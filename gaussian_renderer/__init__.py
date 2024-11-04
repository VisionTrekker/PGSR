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

import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    """
    从渲染的 相机坐标系下的深度图 计算 相机坐标系下的法向量
        viewpoint_cam：当前相机
        depth:  相机坐标系下的无偏深度图，(H, W)
        offset: 每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度
    """
    # bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)   # 获取当前相机的 内参矩阵(C2pix) 和 外参矩阵(W2C)
    st = max(int(scale / 2) - 1, 0) # 如果scale>2，则st为(scale/2)-1的向下取整；否则为0
    if offset is not None:
        offset = offset[st::scale,st::scale]    # 如果输入了偏移量，也对其进行采样（减少计算量，并且采样时丢弃初始的行和列避免边缘的影响），采样后的大小为(H-st)//scale
    # 从相机坐标系下的深度图 计算法向量（相机坐标系）
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)  # (C,H,W)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, return_plane = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        # < 7000代，则执行下面的渲染流程（不渲染法向量）后，直接返回
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        if app_model is not None and pc.use_app:
            # > 1000代 且 开启曝光补偿，则使用app_model生成app_image
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    # > 7000代才返回normal
    global_normal = pc.get_normal(viewpoint_camera) # 获取世界坐标系下所有高斯的法向量，即最短轴向量
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3] # 当前相机坐标系下 所有高斯的法向量

    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]   # 当前相机坐标系下所有高斯中心的坐标
    depth_z = pts_in_cam[:, 2]  # 所有高斯在当前相机坐标系下的深度

    local_distance = (local_normal * pts_in_cam).sum(-1).abs()  # 相机光心 到 所有高斯法向量垂直平面的 距离 = 相机光心 与 所有高斯中心 投影到高斯法向量方向上的 距离

    # (N, 5)，依次存储：[N, 0-2] 当前相机坐标系下所有高斯的法向量，即最短轴向量；[N,3] 全1.0；[N,4] 相机光心 到 所有高斯法向量垂直平面的 距离
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    # 返回：
    #   渲染的 RGB图像
    #   所有高斯投影在当前相机图像平面上的最大半径 数组
    #   所有高斯 渲染时在透射率>0.5之前 对某像素有贡献的 像素个数 数组
    #   5通道tensor，[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
    #   渲染的 无偏深度图（相机坐标系）
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,    # 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]      # 渲染的 法向量（相机坐标系）
    rendered_alpha = out_all_map[3:4, ]     # 每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度
    rendered_distance = out_all_map[4:5, ]  # 渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,    # 所有高斯投影在当前相机图像平面上的最大半径>0的mask，(N,)
                    "radii": radii,
                    "out_observe": out_observe,     # 所有高斯 渲染时在透射率>0.5之前 对某像素有贡献的 像素个数，(N,)
                    "rendered_normal": rendered_normal, # 渲染的 法向量（相机坐标系）
                    "plane_depth": plane_depth,     # 渲染的 无偏深度图（相机坐标系）
                    "rendered_distance": rendered_distance  # 渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
                    }
    
    if app_model is not None and pc.use_app:
        # > 1000代 且 开启曝光补偿，则使用app_model生成app_image
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    if return_depth_normal:
        # > 7000代，返回从渲染深度图计算的 法向量（相机坐标系）
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict