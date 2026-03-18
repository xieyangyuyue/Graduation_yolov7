import cv2
import numpy as np
import os
import shutil
from glob import glob
from tqdm import tqdm

def enhance_image(image_path, output_path, gamma=0.6, clip_limit=2.0, tile_grid_size=(8, 8)):
    """对单张图像进行 Gamma + CLAHE 增强"""
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 1. Gamma 校正 (全局提亮)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # 2. CLAHE 局部对比度增强
    lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # 3. 合并通道并保存
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, final_img)
    return True

def process_yolo_dataset(input_dir, output_dir):
    """遍历 YOLO 格式的数据集，增强图像并同步复制标签"""
    splits = ['train', 'val', 'test']

    for split in splits:
        in_split_dir = os.path.join(input_dir, split)
        if not os.path.exists(in_split_dir):
            print(f"未找到 {split} 文件夹，跳过...")
            continue

        # 创建输出子目录
        out_split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(out_split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_split_dir, 'labels'), exist_ok=True)

        # 1. 处理并增强图像
        img_paths = glob(os.path.join(in_split_dir, 'images', '*.*'))
        print(f"\n--- 正在处理 {split} 集图像 (共 {len(img_paths)} 张) ---")
        for img_path in tqdm(img_paths, desc="Enhancing Images"):
            filename = os.path.basename(img_path)
            out_img_path = os.path.join(out_split_dir, 'images', filename)
            enhance_image(img_path, out_img_path)

        # 2. 原封不动地复制标签文件
        label_paths = glob(os.path.join(in_split_dir, 'labels', '*.txt'))
        print(f"--- 正在同步 {split} 集标签 (共 {len(label_paths)} 个) ---")
        for label_path in tqdm(label_paths, desc="Copying Labels"):
            filename = os.path.basename(label_path)
            out_label_path = os.path.join(out_split_dir, 'labels', filename)
            shutil.copy(label_path, out_label_path)

if __name__ == '__main__':
    # 你的 YOLO 格式原始数据集路径 (参考你 ExDark.yaml 的路径)
    INPUT_DIR = './datasets/ExDark'
    # 增强后的新数据集输出路径
    OUTPUT_DIR = './datasets/ExDark_Enhanced'

    print("开始进行 CLAHE + Gamma 数据集离线增强...")
    process_yolo_dataset(INPUT_DIR, OUTPUT_DIR)
    print("\n✅ 全部处理完成！你的增强数据集已准备就绪，标签已完美对齐！")
