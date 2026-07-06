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
import math
import torch
from random import randint
from utils.loss_utils import *
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, GradientScaler
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2_cu
from utils.lie_groups import exp_map_SO3xR3
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
ENABLE_PROGRESS_PLOT = False

from color_mlp import ColorMLP
from depth_mlp import DepthMLP


def _cov6_to_mat3(cov6):
    cov = torch.zeros((cov6.shape[0], 3, 3), dtype=cov6.dtype, device=cov6.device)
    cov[:, 0, 0] = cov6[:, 0]
    cov[:, 0, 1] = cov[:, 1, 0] = cov6[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = cov6[:, 2]
    cov[:, 1, 1] = cov6[:, 3]
    cov[:, 1, 2] = cov[:, 2, 1] = cov6[:, 4]
    cov[:, 2, 2] = cov6[:, 5]
    return cov


def camera_projection_bridge_3dgs(
    viewpoint_cam,
    means3D,
    viewspace_point_tensor,
    conic_grad_holder=None,
    cov3D_precomp=None,
    low_pass_filter_radius=0.3,
):
    if viewspace_point_tensor is None or viewspace_point_tensor.grad is None:
        return False
    if not hasattr(viewpoint_cam, "cam_pose_adj") or not viewpoint_cam.cam_pose_adj.requires_grad:
        return False

    dL_dmean2D = viewspace_point_tensor.grad.detach()
    if dL_dmean2D.ndim == 3:
        dL_dmean2D = dL_dmean2D.squeeze(0)
    if dL_dmean2D.shape[-1] < 2:
        return False
    dL_dmean2D = dL_dmean2D[:, :2]
    if not torch.isfinite(dL_dmean2D).all():
        return False
    if float(dL_dmean2D.abs().max().item()) < 1e-12:
        has_mean_grad = False
    else:
        has_mean_grad = True

    dL_dconic = None
    if (
        conic_grad_holder is not None
        and conic_grad_holder.grad is not None
        and cov3D_precomp is not None
        and cov3D_precomp.numel() > 0
    ):
        raw_dL_dconic = conic_grad_holder.grad.detach()
        if raw_dL_dconic.ndim == 3 and raw_dL_dconic.shape[-2:] == (2, 2):
            dL_dconic = torch.stack(
                [
                    raw_dL_dconic[:, 0, 0],
                    raw_dL_dconic[:, 0, 1],
                    raw_dL_dconic[:, 1, 1],
                ],
                dim=-1,
            )
            if (not torch.isfinite(dL_dconic).all()) or float(dL_dconic.abs().max().item()) < 1e-12:
                dL_dconic = None
    if not has_mean_grad and dL_dconic is None:
        return False

    cam_opt_mat = exp_map_SO3xR3(viewpoint_cam.cam_pose_adj)
    dR = cam_opt_mat[0, :3, :3]
    dt = cam_opt_mat[0, :3, 3]
    R = viewpoint_cam.R_cu.matmul(dR.T)
    T = dt + dR.matmul(viewpoint_cam.T_cu)
    world_view_transform = getWorld2View2_cu(
        R, T, viewpoint_cam.trans_cu, viewpoint_cam.scale_cu
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)

    means_h = torch.cat([means3D.detach(), torch.ones_like(means3D[:, :1])], dim=-1)
    clip = means_h @ full_proj_transform
    ndc_xy = clip[:, :2] / (clip[:, 3:4] + 1e-7)
    # RNG rasterizes on the uncropped full canvas and crops the final image
    # afterwards, so CUDA's dL/dmean2D lives in full-canvas pixel coordinates.
    screen_width = int(getattr(viewpoint_cam, "full_width", viewpoint_cam.image_width))
    screen_height = int(getattr(viewpoint_cam, "full_height", viewpoint_cam.image_height))
    pixel_xy = torch.empty_like(ndc_xy)
    pixel_xy[:, 0] = ((ndc_xy[:, 0] + 1.0) * screen_width - 1.0) * 0.5
    pixel_xy[:, 1] = ((ndc_xy[:, 1] + 1.0) * screen_height - 1.0) * 0.5
    if not torch.isfinite(pixel_xy).all():
        return False

    proxy = pixel_xy.new_zeros(())
    if has_mean_grad:
        proxy = proxy + (pixel_xy * dL_dmean2D).sum()

    if dL_dconic is not None:
        cov3D = _cov6_to_mat3(cov3D_precomp.detach())
        tan_fovx = math.tan(viewpoint_cam.FoVx * 0.5)
        tan_fovy = math.tan(viewpoint_cam.FoVy * 0.5)
        focal_x = float(screen_width) / (2.0 * tan_fovx)
        focal_y = float(screen_height) / (2.0 * tan_fovy)

        t = means_h @ world_view_transform
        tz = t[:, 2].clamp_min(1e-7)
        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        tx = torch.clamp(t[:, 0] / tz, min=-limx, max=limx) * tz
        ty = torch.clamp(t[:, 1] / tz, min=-limy, max=limy) * tz
        tz2 = tz * tz

        J = torch.zeros((means3D.shape[0], 3, 3), dtype=means3D.dtype, device=means3D.device)
        J[:, 0, 0] = focal_x / tz
        J[:, 1, 1] = focal_y / tz
        J[:, 2, 0] = -(focal_x * tx) / tz2
        J[:, 2, 1] = -(focal_y * ty) / tz2

        W = world_view_transform[:3, :3]
        Tmat = torch.matmul(W.unsqueeze(0), J)
        cov2D = torch.matmul(torch.matmul(Tmat.transpose(1, 2), cov3D.transpose(1, 2)), Tmat)
        a = cov2D[:, 0, 0] + low_pass_filter_radius
        b = cov2D[:, 0, 1]
        c = cov2D[:, 1, 1] + low_pass_filter_radius
        det = a * c - b * b
        denom2inv = 1.0 / (det * det + 1e-7)
        valid = torch.isfinite(denom2inv) & (denom2inv != 0)
        valid = valid & torch.isfinite(dL_dconic).all(dim=-1)
        valid = valid & (dL_dconic.abs().sum(dim=-1) > 1e-12)
        if bool(valid.any().item()):
            dconic = dL_dconic[valid]
            av = a[valid]
            bv = b[valid]
            cv = c[valid]
            detv = det[valid]
            denom2inv_v = denom2inv[valid]
            dL_da = denom2inv_v * (
                -cv * cv * dconic[:, 0]
                + 2.0 * bv * cv * dconic[:, 1]
                + (detv - av * cv) * dconic[:, 2]
            )
            dL_dc = denom2inv_v * (
                -av * av * dconic[:, 2]
                + 2.0 * av * bv * dconic[:, 1]
                + (detv - av * cv) * dconic[:, 0]
            )
            dL_db = denom2inv_v * 2.0 * (
                bv * cv * dconic[:, 0]
                - (detv + 2.0 * bv * bv) * dconic[:, 1]
                + av * bv * dconic[:, 2]
            )
            cov_proxy = av * dL_da.detach() + bv * dL_db.detach() + cv * dL_dc.detach()
            proxy = proxy + cov_proxy.sum()

    if not torch.isfinite(proxy):
        return False
    proxy.backward()
    return True


def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, pc_ckpt, mlp_ckpt, d_mlp_ckpt, loss_type, debug_from, crop_pc=0.0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    from datetime import datetime
    log_txt_file_name = f"progress-{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
    with open(os.path.join(dataset.model_path, log_txt_file_name), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=opt, load_valid=False)
    if pc_ckpt:
        (model_params, first_iter) = torch.load(pc_ckpt, map_location='cuda', weights_only=False)
        gaussians.restore(model_params, opt)
        print('Loaded point cloud ({} pts) from checkpoint. Iteration: {}'.format(gaussians.get_xyz.shape[0], first_iter))
        if scene.optimizing_cameras:
            scene.load_camera_adjustments(first_iter)
        if pipe.reset_features:
            gaussians.reset_features()
            print('Features are reset after loading point cloud.')
    gaussians.training_setup(opt)
    ## explicitly crop the point cloud after possibly loading and training setup
    gaussians.crop_pc(crop_pc)
    print(f'Total number of gaussians after training setup: {scene.gaussians.get_xyz.shape[0]}')
        
    ## initialize color mlp
    color_mlp = None
    if pipe.color_mlp:
        in_channels = pipe.in_channels + 6 + (pipe.encoding_levels_each * 12 if pipe.encoding_levels_each > 0 else 0) + 1 ## pl_distance
        if pipe.shadow_map:
            in_channels += 1 + (pipe.encoding_levels_shadow * 2 if pipe.encoding_levels_shadow > 0 else 0)
        print(f'in channels: {in_channels}')
        color_mlp = ColorMLP(in_channels=in_channels, checkpoint=mlp_ckpt).cuda()
    
    depth_mlp = None
    if pipe.depth_mlp:
        depth_mlp = DepthMLP(in_channels=3, checkpoint=d_mlp_ckpt, depth_mlp_modifier=pipe.depth_mlp_modifier).cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).cuda()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loss_func = eval(f'{loss_type}_loss')
    viewpoint_stack = None
    opt_test = False
    opt_test_ready = False
    ema_loss_for_log = 0.0
    loss_curve = []
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera.  When camera optimization is enabled, alternate a
        # full train-camera pass with a full test-camera pose-only pass, matching
        # the GS^3 evaluation protocol without updating scene parameters on test.
        if not viewpoint_stack:
            if opt_test_ready and scene.optimizing_cameras and len(scene.getTestCameras()) > 0:
                opt_test = True
                viewpoint_stack = scene.getTestCameras().copy()
                opt_test_ready = False
            else:
                opt_test = False
                viewpoint_stack = scene.getTrainCameras().copy()
                opt_test_ready = True
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        if scene.optimizing_cameras:
            scene.update_camera_learning_rate(iteration)
            viewpoint_cam.update("SO3xR3", update_rays=(pipe.defer_shading or pipe.shadow_map))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, color_mlp = color_mlp, depth_mlp = depth_mlp, iteration=iteration)
        image, depth, alpha, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg['depth'], render_pkg['alpha'], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if depth_mlp is not None:
            depth_offsets = render_pkg['depth_offset'].mean().detach().cpu().numpy()
        if pipe.gradient_scaling:
            image, alpha, depth = GradientScaler.apply(image, alpha, depth)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() ## [3, H, W]
        if dataset.hdr:
            if iteration == 1:
                print(
                    f"[HDR_GAMMA] enabled freeze={opt.hdr_gamma_freeze_step} "
                    f"fit={opt.hdr_gamma_fit_step}"
                )
            if iteration <= opt.hdr_gamma_freeze_step:
                gt_image = torch.pow(gt_image, 1.0 / 2.2)
            elif iteration < opt.hdr_gamma_freeze_step + opt.hdr_gamma_fit_step // 2:
                gamma = 1.1 * float(
                    opt.hdr_gamma_freeze_step + opt.hdr_gamma_fit_step - iteration + 1
                ) / float(opt.hdr_gamma_fit_step // 2 + 1)
                gt_image = torch.pow(gt_image, 1.0 / gamma)
        loss1 = loss_func(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * loss1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if scene.optimizing_cameras and opt.cam_reg > 0:
            loss = loss + opt.cam_reg * viewpoint_cam.get_loss()
        loss.backward(retain_graph=True)
        if scene.optimizing_cameras and not opt.disable_cam_bridge:
            camera_projection_bridge_3dgs(
                viewpoint_cam,
                gaussians.get_xyz,
                viewspace_point_tensor,
                conic_grad_holder=render_pkg.get("conic_grad_holder"),
                cov3D_precomp=gaussians.get_covariance().detach(),
            )

        if iteration == 1:
            print(image.shape)
            print(gt_image.shape)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                line = f"Training Loss={ema_loss_for_log:>8.5f} #G={scene.gaussians.get_xyz.shape[0]:>7d}"
                if depth_mlp is not None:
                    line += f" ΔD={depth_offsets:>7.3f}"
                if scene.optimizing_cameras:
                    cam_norm = viewpoint_cam.cam_pose_adj.detach().norm().item()
                    cam_grad = viewpoint_cam.cam_pose_adj.grad
                    cam_grad_norm = cam_grad.detach().norm().item() if cam_grad is not None else 0.0
                    cam_lr = 0.0
                    if scene.camera_optimizer is not None:
                        cam_lr = scene.camera_optimizer.param_groups[0].get("lr", 0.0)
                    split_name = "test" if opt_test else "train"
                    line += f" split={split_name} cam={cam_norm:.2e} camG={cam_grad_norm:.2e} camLR={cam_lr:.2e}"
                progress_bar.set_description(line)
                with open(os.path.join(dataset.model_path, log_txt_file_name), 'a') as f:
                    f.write(f'{datetime.now().strftime("%H:%M:%S")} {line}\n')
                progress_bar.update(10)
                loss_curve.append(loss.item())
            if iteration == opt.iterations:
                progress_bar.close()
                
            # Log and save
            training_report(tb_writer, iteration, loss1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), color_mlp)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if color_mlp is not None:
                    color_mlp_save_dir = os.path.join(dataset.model_path, 'color_mlp', f'iteration_{iteration}')
                    os.makedirs(color_mlp_save_dir, exist_ok=True)
                    torch.save((color_mlp.capture(), iteration), os.path.join(color_mlp_save_dir, f'color_mlp_chkpnt{iteration}.pth'))
                if depth_mlp is not None:
                    depth_mlp_save_dir = os.path.join(dataset.model_path, 'depth_mlp', f'iteration_{iteration}')
                    os.makedirs(depth_mlp_save_dir, exist_ok=True)
                    torch.save((depth_mlp.capture(), iteration), os.path.join(depth_mlp_save_dir, f'depth_mlp_chkpnt{iteration}.pth'))

            if opt_test and scene.optimizing_cameras:
                # Test images are used only to refine their camera poses.  Do not
                # update Gaussians or MLPs from held-out images.
                if iteration < opt.iterations:
                    scene.step_camera_optimizer()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if color_mlp is not None:
                        color_mlp.optimizer.zero_grad(set_to_none=True)
                    if depth_mlp is not None:
                        depth_mlp.optimizer.zero_grad(set_to_none=True)
                    viewpoint_cam.update("SO3xR3", update_rays=(pipe.defer_shading or pipe.shadow_map))
            else:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_min_opacity, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                        gaussians.crop_pc(crop_pc)
                        torch.cuda.empty_cache()
                        print("Number of Gaussians: {}".format(scene.gaussians.get_xyz.shape[0]))

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if color_mlp is not None:
                        color_mlp.optimizer.step()
                        color_mlp.optimizer.zero_grad(set_to_none = True)
                    if depth_mlp is not None:
                        depth_mlp.optimizer.step()
                        depth_mlp.optimizer.zero_grad(set_to_none = True)
                    scene.step_camera_optimizer()
                    if scene.optimizing_cameras:
                        viewpoint_cam.update("SO3xR3", update_rays=(pipe.defer_shading or pipe.shadow_map))

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if color_mlp is not None:
                    torch.save((color_mlp.capture(), iteration), scene.model_path + "/color_mlp_chkpnt" + str(iteration) + ".pth")
                if depth_mlp is not None:
                    torch.save((depth_mlp.capture(), iteration), scene.model_path + "/depth_mlp_chkpnt" + str(iteration) + ".pth")
                    
            if iteration % 15000 == 0:
                for gidx, pgroup in enumerate(gaussians.optimizer.param_groups):
                    pgroup['lr'] *= 0.75
                    print(f"[ITER {iteration}] gaussian {gidx} lr -> {pgroup['lr']:.2e}")
                if color_mlp is not None:
                    color_mlp.optimizer.param_groups[0]['lr'] *= 0.75
                    print(f"[ITER {iteration}] color_mlp lr -> {color_mlp.optimizer.param_groups[0]['lr']:.2e}")
                if depth_mlp is not None:
                    depth_mlp.optimizer.param_groups[0]['lr'] *= 0.75
                    print(f"[ITER {iteration}] depth_mlp lr -> {depth_mlp.optimizer.param_groups[0]['lr']:.2e}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if False:# TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, color_mlp=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, color_mlp=color_mlp)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_pc", type=str, default = None)
    parser.add_argument("--load_mlp", type=str, default = None)
    parser.add_argument("--load_d_mlp", type=str, default = None)
    parser.add_argument("--loss", type=str, default="l1", choices=['l1', 'l2', 'logl1', 'logl2'])
    parser.add_argument("--crop_pc", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations = args.save_iterations
    
    ## save cmd-line args to txt file
    os.makedirs(args.model_path, exist_ok=True)
            
    ## recursively backup all python scripts
    os.makedirs(os.path.join(args.model_path, "codes"), exist_ok=True)
    for root, dirs, files in os.walk("/home/lab409/3dgs-pl/gaussian-splatting", topdown=True):
        for file in files:
            if file.endswith(".py"):
                os.makedirs(root.replace("/home/lab409/3dgs-pl/gaussian-splatting", os.path.join(args.model_path, "codes")), exist_ok=True)
                # os.system(f"cp {os.path.join(root, file)} {os.path.join(args.model_path, 'codes', root.split('/')[-1])}")
                os.system(f'cp {os.path.join(root, file)} {os.path.join(root.replace("/home/lab409/3dgs-pl/gaussian-splatting", os.path.join(args.model_path, "codes")), file)}')
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.load_pc, args.load_mlp, args.load_d_mlp, args.loss, args.debug_from, args.crop_pc)

    # All done
    print("Training complete.")
