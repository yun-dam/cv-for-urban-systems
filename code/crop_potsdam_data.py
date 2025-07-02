import os
import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# 配置 (Configuration)
# ==============================================================================
# 原始数据目录
# ORIGINAL_IMAGE_DIR = "./data/Vaihingen/top"
# ORIGINAL_LABEL_DIR = "./data/Vaihingen/ground_truth"
ORIGINAL_IMAGE_DIR = "./data/Potsdam/top"
ORIGINAL_LABEL_DIR = "./data/Potsdam/ground_truth"

# 裁剪后数据的输出目录
# CROPPED_IMAGE_DIR = "./data/Vaihingen/top_cropped_512"
# CROPPED_LABEL_DIR = "./data/Vaihingen/ground_truth_cropped_512"
CROPPED_IMAGE_DIR = "./data/Potsdam/top_cropped_512"
CROPPED_LABEL_DIR = "./data/Potsdam/ground_truth_cropped_512"

# 定义图块大小和步长
PATCH_SIZE = 512
STRIDE = 512 # 使用与PATCH_SIZE相同的值表示无重叠裁剪

# ==============================================================================
# 标签文件名映射函数
# ==============================================================================
def get_label_filename(image_filename):
    """
    将top_potsdam_2_10_RGB.tif映射为top_potsdam_2_10_label.tif
    """
    return image_filename.replace('_RGB', '_label')

# ==============================================================================
# 主函数 (Main Function)
# ==============================================================================
def crop_dataset():
    """
    遍历原始图像和标签，将它们裁剪成指定大小的图块并保存。
    """
    print("开始裁剪数据集...")
    
    # 创建输出目录
    os.makedirs(CROPPED_IMAGE_DIR, exist_ok=True)
    os.makedirs(CROPPED_LABEL_DIR, exist_ok=True)

    image_files = sorted([f for f in os.listdir(ORIGINAL_IMAGE_DIR) if f.endswith(".tif")])

    total_patches = 0
    for image_file in tqdm(image_files, desc="Cropping files"):
        # 构建路径
        image_path = os.path.join(ORIGINAL_IMAGE_DIR, image_file)
        # 使用自定义函数获取标签文件名
        label_file = get_label_filename(image_file)
        label_path = os.path.join(ORIGINAL_LABEL_DIR, label_file)

        if not os.path.exists(label_path):
            print(f"警告: 找不到对应的标签文件 {label_path}，跳过 {image_file}")
            continue

        # 读取图像和标签
        try:
            image = tifffile.imread(image_path)
            label = tifffile.imread(label_path)
        except Exception as e:
            print(f"错误：读取文件 {image_file} 或 {label_file} 时出错: {e}")
            continue
        
        # 验证图像和标签尺寸是否一致
        if image.shape[:2] != label.shape[:2]:
            print(f"警告: 图像和标签尺寸不匹配 {image_file}，跳过。")
            continue

        img_h, img_w = image.shape[:2]

        # 使用滑动窗口进行裁剪
        for y in range(0, img_h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, img_w - PATCH_SIZE + 1, STRIDE):
                # 裁剪图像和标签
                image_patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, ...]
                label_patch = label[y:y + PATCH_SIZE, x:x + PATCH_SIZE, ...]

                # 构建输出文件名
                base_name = Path(image_file).stem
                patch_filename = f"{base_name}_patch_{y}_{x}.tif"

                # 保存图块
                tifffile.imwrite(os.path.join(CROPPED_IMAGE_DIR, patch_filename), image_patch)
                tifffile.imwrite(os.path.join(CROPPED_LABEL_DIR, patch_filename), label_patch)

                total_patches += 1

    print(f"\n裁剪完成！总共生成了 {total_patches} 个图块。")
    print(f"图像图块保存在: {CROPPED_IMAGE_DIR}")
    print(f"标签图块保存在: {CROPPED_LABEL_DIR}")

if __name__ == "__main__":
    crop_dataset()