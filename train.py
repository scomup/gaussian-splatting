#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import numpy as np
import gsplatcu as gsc


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw


class GSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pws,
        shs,
        alphas,
        scales,
        rots,
        us,
        cam,
    ):
        # more detail view forward.pdf
        # step1. Transform pw to camera frame,
        # and project it to iamge.
        us, pcs, depths, du_dpcs = gsc.project(
            pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, True)

        # step2. Calcuate the 3d Gaussian.
        cov3ds, dcov3d_drots, dcov3d_dscales = gsc.computeCov3D(
            rots, scales, depths, True)

        # step3. Calcuate the 2d Gaussian.
        cov2ds, dcov2d_dcov3ds, dcov2d_dpcs = gsc.computeCov2D(
            cov3ds, pcs, cam.Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, True)

        # step4. get color info
        colors, dcolor_dshs, dcolor_dpws = gsc.sh2Color(shs.reshape(shs.shape[0], -1), pws, cam.twc, True)

        # step5. Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = gsc.inverseCov2D(cov2ds, depths, True)
        image, contrib, final_tau, patch_range_per_tile, gsid_per_patch =\
            gsc.splat(cam.height, cam.width,
                      us, cinv2ds, alphas, depths, colors, areas)

        # Store the static parameters in the context
        ctx.cam = cam
        # Keep relevant tensors for backward
        ctx.save_for_backward(us, cinv2ds, alphas,
                              depths, colors, contrib, final_tau,
                              patch_range_per_tile, gsid_per_patch,
                              dcinv2d_dcov2ds, dcov2d_dcov3ds,
                              dcov3d_drots, dcov3d_dscales, dcolor_dshs,
                              du_dpcs, dcov2d_dpcs, dcolor_dpws)
        return image, areas

    @staticmethod
    def backward(ctx, dloss_dgammas, _):
        # Retrieve the saved tensors and static parameters
        cam = ctx.cam
        us, cinv2ds, alphas, \
            depths, colors, contrib, final_tau,\
            patch_range_per_tile, gsid_per_patch,\
            dcinv2d_dcov2ds, dcov2d_dcov3ds,\
            dcov3d_drots, dcov3d_dscales, dcolor_dshs,\
            du_dpcs, dcov2d_dpcs, dcolor_dpws = ctx.saved_tensors

        # more detail view backward.pdf

        # section.5
        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(cam.height, cam.width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_range_per_tile, gsid_per_patch, dloss_dgammas)

        dpc_dpws = cam.Rcw
        dloss_dcov2ds = dloss_dcinv2ds @ dcinv2d_dcov2ds
        # backward.pdf equation (3)
        dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
        # backward.pdf equation (4)
        dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
        # backward.pdf equation (5)
        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @
                      dcolor_dshs).permute(0, 2, 1).squeeze()

        # dloss_dshs = dloss_dshs.reshape(dloss_dshs.shape[0], -1)
        # backward.pdf equation (7)
        dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
            dloss_dcolors @ dcolor_dpws + \
            dloss_dcov2ds @ dcov2d_dpcs @ dpc_dpws

        new_dloss_dus = torch.zeros(dloss_dus.shape[0], 3).to(torch.float32).to('cuda')

        # Assuming your original tensor has shape [N, 2]
        # Copy the data from the original tensor to the new tensor
        new_dloss_dus[:, 0] = dloss_dus[:, 0, 0] * cam.width * 0.5
        new_dloss_dus[:, 1] = dloss_dus[:, 0, 1] * cam.height * 0.5

        return dloss_dpws.squeeze(),\
            dloss_dshs.squeeze(),\
            dloss_dalphas.squeeze().unsqueeze(1),\
            dloss_dscales.squeeze(),\
            dloss_drots.squeeze(),\
            new_dloss_dus.squeeze(),\
            None


fig, ax = plt.subplots()
array = np.zeros(shape=(545, 980, 3), dtype=np.uint8)
im = ax.imshow(array)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        fx = viewpoint_cam.image_width / (2 * np.tan(viewpoint_cam.FoVx / 2))
        fy = viewpoint_cam.image_height / (2 * np.tan(viewpoint_cam.FoVy / 2))
        cx = viewpoint_cam.image_width / 2
        cy = viewpoint_cam.image_height / 2
        Rcw = torch.from_numpy(viewpoint_cam.R.transpose()).to(torch.float32).to('cuda')
        tcw = torch.from_numpy(viewpoint_cam.T).to(torch.float32).to('cuda')

        cam = Camera(viewpoint_cam.colmap_id, viewpoint_cam.image_width,
                     viewpoint_cam.image_height, fx, fy, cx, cy, Rcw, tcw)

        means3D = gaussians.get_xyz
        # means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        viewspace_point_tensor = torch.zeros(
            [means3D.shape[0], 3], dtype=torch.float32, device='cuda', requires_grad=True)

        opacity = gaussians.get_opacity

        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        shs = gaussians.get_features

        image, areas = GSFunction.apply(means3D, shs, opacity, scales, rotations, viewspace_point_tensor, cam)
        radii = torch.norm(areas.to(torch.float32),dim=1)
        visibility_filter = radii > 0
        # viewspace_point_tensor = means2D

        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
        #     "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # viewpoint_cam.colmap_id

        if (viewpoint_cam.uid == 1):
            im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
            im.set_data(im_cpu)
            plt.pause(0.1)

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    #print(args)
    #exit()
    args.source_path='/home/liu/bag/gaussian-splatting/tandt/train'
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")