import os

gpu_id = 2

# DTU数据集：-r2 --ncc_scale 0.5
# MipNeRF360数据集：
#   室外场景 bicycle, flowers, garden, stump, treehill： -r 4 --ncc_scale 0.5
#   室内场景 bonsai, counter, kitchen, room：            -r 2 --ncc_scale 0.5
# TnT数据集：-r2 --ncc_scale 0.5 --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation


print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s /data2/jtx/data/popmart/new -m ./output/popmart -r 2 --ncc_scale 0.5'
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py \
        -s /data2/liuzhi/Dataset/3DGS_Dataset/input/gm_Museum \
        -m ./output/gm_Museum \
        -r 2 \
        --ncc_scale 0.5 \
        --densify_abs_grad_threshold 0.0004'
# print(cmd)
# os.system(cmd)

print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m ./output/popmart'
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py \
        -m ./output/gm_Museum'
print(cmd)
os.system(cmd)