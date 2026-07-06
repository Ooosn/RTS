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
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(
        self,
        args : ModelParams,
        gaussians : GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        crop=0.0,
        opt=None,
        load_valid=True,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.optimizing_cameras = False
        self.camera_optimizer = None
        self.camera_scheduler_args = None
        self.train_cam_freeze_step = 0

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.valid_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")) and not args.json:
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.less, args.max_training_images, '.png', args.max_reso)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")) and args.json:
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                args.less,
                args.max_training_images,
                '.exr' if args.hdr else '.png',
                args.max_reso,
                load_valid=load_valid,
                load_train=getattr(args, "_render_load_train", True),
                load_test=getattr(args, "_render_load_test", True),
            )
        else:
            assert False, "Could not recognize scene type!"
        if not load_valid:
            scene_info = scene_info._replace(valid_cameras=[])
        print(f'Loaded {len(scene_info.train_cameras)} train images.')
        print(f'Loaded {len(scene_info.test_cameras)} test images.')
        print(f'Loaded {len(scene_info.valid_cameras or [])} valid cameras.')

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.valid_cameras:
                camlist.extend(scene_info.valid_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            if scene_info.valid_cameras:
                random.shuffle(scene_info.valid_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Valid Cameras")
            self.valid_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.valid_cameras or [], resolution_scale, args)

        if self.loaded_iter:
            print("Loading point cloud from iteration {}".format(self.loaded_iter))
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.load_camera_adjustments(self.loaded_iter)
        else:
            print("Creating point cloud from scene")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        self.gaussians.crop_pc(crop)

        if opt is not None and getattr(opt, "cam_opt", False):
            cam_params = []
            self.optimizing_cameras = True
            self.train_cam_freeze_step = int(getattr(opt, "train_cam_freeze_step", 0))
            self.camera_scheduler_args = get_expon_lr_func(
                lr_init=opt.cam_lr_init,
                lr_final=opt.cam_lr_final,
                lr_delay_steps=opt.cam_lr_delay_steps,
                lr_delay_mult=opt.cam_lr_delay_mult,
                max_steps=opt.cam_lr_max_steps,
            )
            for scale in resolution_scales:
                for cam in self.train_cameras[scale]:
                    cam.cam_pose_adj.requires_grad_(True)
                    cam_params.append(cam.cam_pose_adj)
                for cam in self.test_cameras[scale]:
                    cam.cam_pose_adj.requires_grad_(True)
                    cam_params.append(cam.cam_pose_adj)
                for cam in self.valid_cameras[scale]:
                    cam.cam_pose_adj.requires_grad_(True)
                    cam_params.append(cam.cam_pose_adj)

            if len(cam_params) > 0:
                self.camera_optimizer = torch.optim.Adam(
                    [{"params": cam_params, "lr": 0.0, "name": "cam_adj"}],
                    lr=0.0,
                    eps=1e-15,
                )
            else:
                self.optimizing_cameras = False

    def update_camera_learning_rate(self, iteration):
        if self.camera_optimizer is None or self.camera_scheduler_args is None:
            return 0.0
        if iteration < self.train_cam_freeze_step:
            lr = 0.0
            for param_group in self.camera_optimizer.param_groups:
                param_group["lr"] = lr
            return lr
        lr = self.camera_scheduler_args(iteration)
        for param_group in self.camera_optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def step_camera_optimizer(self):
        if self.camera_optimizer is None:
            return
        self.camera_optimizer.step()
        self.camera_optimizer.zero_grad(set_to_none=True)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.save_camera_adjustments(iteration)

    def _camera_adjustment_path(self, iteration):
        return os.path.join(
            self.model_path,
            "point_cloud",
            "iteration_{}".format(iteration),
            "camera_adjustments.pt",
        )

    def save_camera_adjustments(self, iteration):
        state = {}
        for split_name, camera_groups in (
            ("train", self.train_cameras),
            ("test", self.test_cameras),
            ("valid", self.valid_cameras),
        ):
            split_state = {}
            for scale, cameras in camera_groups.items():
                split_state[str(scale)] = [
                    {
                        "image_name": str(cam.image_name),
                        "cam_pose_adj": cam.cam_pose_adj.detach().cpu(),
                    }
                    for cam in cameras
                    if hasattr(cam, "cam_pose_adj")
                ]
            state[split_name] = split_state
        torch.save(state, self._camera_adjustment_path(iteration))

    def load_camera_adjustments(self, iteration):
        path = self._camera_adjustment_path(iteration)
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location="cpu")
        restored = 0
        for split_name, camera_groups in (
            ("train", self.train_cameras),
            ("test", self.test_cameras),
            ("valid", self.valid_cameras),
        ):
            split_state = state.get(split_name, {}) if isinstance(state, dict) else {}
            for scale, cameras in camera_groups.items():
                saved_list = split_state.get(str(scale), [])
                saved_by_name = {
                    str(item.get("image_name")): item
                    for item in saved_list
                    if isinstance(item, dict)
                }
                for cam in cameras:
                    saved = saved_by_name.get(str(cam.image_name))
                    if saved is None or "cam_pose_adj" not in saved:
                        continue
                    with torch.no_grad():
                        cam.cam_pose_adj.copy_(
                            saved["cam_pose_adj"].to(
                                device=cam.cam_pose_adj.device,
                                dtype=cam.cam_pose_adj.dtype,
                            )
                        )
                    cam.update("SO3xR3", update_rays=False)
                    restored += 1
        print(f"Loaded optimized camera adjustments from {path} ({restored})")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getValidCameras(self, scale=1.0):
        return self.valid_cameras[scale]
