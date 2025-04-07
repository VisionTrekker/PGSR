import os
# GPU_ID = 2

############# train参数设置 #############
# 根据数据集调整ModelParams中选择最近邻帧的阈值
# --max_abs_split_points：   对于纹理较弱的场景，为了防止在纹理较弱的区域过度拟合，我们建议禁用这种分割策略，将其设置为0，默认为50_000
# --opacity_cull_threshold：为简单地减少高斯的数量，可将阈值设为0.05，默认为0.005
# DTU数据集：-r2 --ncc_scale 0.5
# MipNeRF360数据集：
#   室外场景 bicycle, flowers, garden, stump, treehill： -r 4 --ncc_scale 0.5 --densify_abs_grad_threshold 0.0004
#   室内场景 bonsai, counter, kitchen, room：            -r 2 --ncc_scale 0.5 --densify_abs_grad_threshold 0.0004
# TnT数据集：-r 2 --ncc_scale 0.5 --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/Dataset/3DGS_Dataset/input/gm_Museum \
#         -m ./output/gm_Museum_0.03gtnormal_new_gtdepth \
#         -r 2 \
#         --ncc_scale 0.5 \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.05 \
#         --exposure_compensation \
#         --load_depth \
#         --load_normal'
# print(cmd)
# os.system(cmd)

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/2floor_multiview_20240905_standard \
#         -m ./output/2floor_multiview_20240905_standard_depth_normal_2 \
#         -r 2 \
#         --data_device "cpu" \
#         --ncc_scale 0.5 \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 60000 \
#         --densify_from_iter 1000 \
#         --densify_until_iter 30000 \
#         --densification_interval 200 \
#         --opacity_reset_interval 6000 \
#         --test_iterations 7000 15000 30000 60000\
# 	    --save_iterations 15000 30000 60000 \
# 	    --load_depth \
# 	    --load_normal \
#         '
# print(cmd)
# os.system(cmd)

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/2floor_multiview_20240905_standard \
#         -m ./output/2floor_multiview_20240905_standard_depth_2 \
#         -r 2 \
#         --data_device "cpu" \
#         --ncc_scale 0.5 \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 60000 \
#         --densify_from_iter 1000 \
#         --densify_until_iter 30000 \
#         --densification_interval 200 \
#         --test_iterations 7000 15000 30000 60000 \
# 	    --save_iterations 15000 30000 60000 \
# 	    --load_depth \
#         '
# print(cmd)
# os.system(cmd)

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/0903_F1yard \
#         -m ./output/siyue_0903_F1yard_depth_normal_onefolder \
#         -r 2 \
#         --data_device "cpu" \
#         --ncc_scale 0.5 \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 60000 \
#         --densify_from_iter 1000 \
#         --densify_until_iter 30000 \
#         --densification_interval 200 \
#         --opacity_reset_interval 6000 \
#         --test_iterations 7000 15000 30000 60000 \
# 	    --save_iterations 15000 30000 60000 \
# 	    --load_depth \
# 	    --load_normal \
#         '
#print(cmd)
#os.system(cmd)
# GPU_ID = 2
# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s ../../remote_data/dataset_reality/anji_qiyu \
#         -m ./output/anji_qiyu_depth \
#         -r 1 \
#         --data_device "cpu" \
#         --ncc_scale 0.5 \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 30000 \
#         --densify_from_iter 500 \
#         --densify_until_iter 15000 \
#         --densification_interval 100 \
#         --opacity_reset_interval 3000 \
#         --single_view_weight_from_iter 7000 \
#         --multi_view_weight_from_iter 30000 \
#         --test_iterations 7000 10000 15000 30000 \
# 	    --save_iterations 7000 10000 15000 30000 \
# 	    --load_depth \
#         '
# print(cmd)
# os.system(cmd)

# GPU_ID = 1
# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s ../../remote_data/dataset_reality/anji_qiyu \
#         -m ./output/anji_qiyu_depth_normal_wo_opacityreset_render \
#         -r 1 \
#         --port 6030 \
#         --data_device "cpu" \
#         --ncc_scale 0.5 \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 30000 \
#         --densify_from_iter 500 \
#         --densify_until_iter 15000 \
#         --densification_interval 100 \
#         --position_lr_init 0.000016 \
#         --opacity_reset_interval 30000 \
#         --single_view_weight_from_iter 7000 \
#         --single_view_weight 0.3 \
#         --multi_view_weight_from_iter 30000 \
#         --test_iterations 7000 10000 15000 30000 \
# 	    --save_iterations 7000 10000 15000 30000 \
# 	    --load_depth \
#         '
# print(cmd)
# os.system(cmd)

GPU_ID = 2
print("----------------------------------------")
cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
        python train.py \
        -s ../../remote_data/dataset_reality/test/t100pro_in_talandB1 \
        -m output/t100pro_in_talandB1_normalGT \
        -r 1 \
        --port 6031 \
        --data_device "cpu" \
        --ncc_scale 0.5 \
        --densify_abs_grad_threshold 0.0004 \
        --opacity_cull_threshold 0.005 \
        --iterations 60000 \
        --split_mode "max_scale" \
        --densify_from_iter 20000 \
        --densify_until_iter 30000 \
        --densification_interval 100 \
        --position_lr_init 0.000016 \
        --opacity_reset_interval 60000 \
        --load_normal \
        --single_view_weight_from_iter 7000 \
        --single_view_weight 0.1 \
        --multi_view_weight_from_iter 60000 \
        --test_iterations 7000 15000 30000 60000 \
	    --save_iterations 30000 60000 \
        '
print(cmd)
os.system(cmd)

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/0909_F1+air \
#         -m ./output/siyue_0909_F1+air_depth_2 \
#         -r 2 \
#         --ncc_scale 0.5 \
#         --data_device "cpu" \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 90000 \
#         --densify_from_iter 1500 \
#         --densify_until_iter 45000 \
#         --densification_interval 300 \
#         --opacity_reset_interval 9000 \
#         --test_iterations 7000 15000 30000 60000 90000 \
# 	    --save_iterations 7000 30000 60000 90000 \
#         --load_depth \
#         '
# print(cmd)
# os.system(cmd)


# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/0910_F1+air_full \
#         -m ./output/0910_F1+air_full_filtered \
#         -r 2 \
#         --ncc_scale 0.5 \
#         --data_device "cpu" \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 300000 \
#         --densify_from_iter 5000 \
#         --densify_until_iter 150000 \
#         --densification_interval 1000 \
#         --opacity_reset_interval 20000 \
#         --test_iterations 30000 60000 90000 150000 200000 300000 \
# 	    --save_iterations 30000 60000 90000 150000 200000 300000 \
# 	    --load_depth \
#         '
# print(cmd)
# os.system(cmd)

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/0910_F1+air_full \
#         -m ./output/0910_F1+air_full_filtered \
#         -r 2 \
#         --ncc_scale 0.5 \
#         --data_device "cpu" \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 300000 \
#         --densify_from_iter 5000 \
#         --densify_until_iter 150000 \
#         --densification_interval 1000 \
#         --opacity_reset_interval 20000 \
#         --test_iterations 30000 60000 90000 150000 200000 300000 \
# 	    --save_iterations 30000 60000 90000 150000 200000 300000 \
# 	    --load_depth \
#         '
# print(cmd)
# os.system(cmd)

# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_reality/siyue/0910_F1+air_full \
#         -m ./output/0910_F1out+air_full_depth_without_multi \
#         -r 2 \
#         --ncc_scale 0.5 \
#         --data_device "cpu" \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0003 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 300000 \
#         --densify_from_iter 5000 \
#         --densify_until_iter 150000 \
#         --densification_interval 1000 \
#         --opacity_reset_interval 30000 \
#         --test_iterations 7000 30000 60000 90000 150000 200000 300000 \
# 	    --save_iterations 30000 120000 200000 300000 \
# 	    --load_depth \
#         '
# print(cmd)
# os.system(cmd)


# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/remote_data/dataset_little/jinhua_data/bajiaotong \
#         -m ./output/bajiaotong \
#         -r 2 \
#         --ncc_scale 0.5 \
#         --data_device "cpu" \
#         --single_view_weight_from_iter 7_000 \
#         --densify_abs_grad_threshold 0.0004 \
#         --opacity_cull_threshold 0.005 \
#         --iterations 30000 \
#         --densify_from_iter 500 \
#         --densify_until_iter 15000 \
#         --densification_interval 100 \
#         --opacity_reset_interval 3000 \
#         --test_iterations 7000 15000 30000 \
# 	    --save_iterations 15000 30000 \
#         '
# print(cmd)
# os.system(cmd)

############# render参数设置 #############
# --use_depth_filter（gm_Museum数据集提取的mesh不好）
# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python render.py \
#         -m ./output/gm_Museum_0.03gtnormal_new_gtdepth \
#         --max_depth 20 \
#         --voxel_size 0.01'
# print(cmd)
# os.system(cmd)