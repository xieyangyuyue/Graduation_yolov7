import cv2
import os
import shutil
from tqdm import tqdm

def enhance_image_clahe(img_path, save_path):
    """使用CLAHE算法增强低光照图像"""
    img = cv2.imread(img_path)
    if img is None:
        return

    # 将图像转换到 LAB 颜色空间 (处理亮度通道 L，保留色彩通道 A 和 B)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建 CLAHE 对象 (clipLimit 控制对比度限制，tileGridSize 控制网格大小)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # 合并通道并转回 BGR
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    cv2.imwrite(save_path, enhanced_img)

def process_dataset(src_dir, dst_dir):
    """复制数据集结构并处理图片"""
    # 1. 复制整个数据集结构（包含 labels 和 images）
    if os.path.exists(dst_dir):
        print(f"目标文件夹 {dst_dir} 已存在，请先删除或重命名。")
        return

    print("正在复制数据集结构和标签...")
    shutil.copytree(src_dir, dst_dir)

    # 2. 遍历新的 images 文件夹，覆盖为增强后的图片
    img_dir = os.path.join(dst_dir, 'images')

    for split in ['train', 'val']:
        split_dir = os.path.join(img_dir, split)
        if not os.path.exists(split_dir): continue

        img_names = os.listdir(split_dir)
        print(f"正在使用 CLAHE 增强 {split} 集图片...")
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(split_dir, img_name)
                # 读取原图，增强后直接覆盖新文件夹里的图
                enhance_image_clahe(img_path, img_path)

    print(f"CLAHE 增强完成！新数据集保存在: {dst_dir}")

if __name__ == "__main__":
    # 【注意】修改为你实际的 YOLO 格式数据集路径
    SOURCE_DATASET = "datasets/ExDark"  # 原数据集
    TARGET_DATASET = "datasets/ExDark_CLAHE" # 增强后的新数据集

    process_dataset(SOURCE_DATASET, TARGET_DATASET)
