import os
import random
from PIL import Image
import argparse

# 标签对应的索引
labels = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

def fix_image_profile(img):
    """转化成RGB格式，避免Libpng警告"""
    try:
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Error fixing color profile: {e}")
        return None

def convert_to_jpg(img_path, output_path, quality=95):
    """统一数据集格式为jpg，可控制质量"""
    try:
        img = Image.open(img_path)
        img = fix_image_profile(img)
        if img is None:
            return None

        jpg_path = os.path.splitext(output_path)[0] + ".jpg"
        img.save(jpg_path, quality=quality, optimize=True)

        if os.path.getsize(jpg_path) == 0:
            print(f"Warning: Empty file created for {img_path}")
            return None

        return jpg_path
    except Exception as e:
        print(f"Error converting {img_path} to JPG: {e}")
        return None

def ExDark2Yolo(txts_dir: str, imgs_dir: str, ratio: str, version: int, output_dir: str, jpg_quality=95, seed=42):
    """改进的数据集转换与划分函数"""

    # 1. 设置随机种子，确保每次划分结果一致（极度重要！）
    random.seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    ratios = [int(r) for r in ratio.split(':')]
    ratio_sum = sum(ratios)
    dataset_perc = {'train': ratios[0] / ratio_sum, 'test': ratios[1] / ratio_sum, 'val': ratios[2] / ratio_sum}

    # 创建子目录
    for t in dataset_perc:
        os.makedirs(os.path.join(output_dir, t, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, t, 'labels'), exist_ok=True)

    total_original_size = 0
    total_converted_size = 0
    processed_count = 0
    skipped_count = 0

    for label in labels:
        print(f'Processing {label}...')
        label_txt_dir = os.path.join(txts_dir, label)

        if not os.path.exists(label_txt_dir):
            print(f"Warning: Label directory {label_txt_dir} does not exist")
            continue

        filenames = os.listdir(label_txt_dir)

        # 2. 核心修复：打乱当前类别的文件列表，确保各类样本在train/val/test中均匀分布
        random.shuffle(filenames)

        files_num = len(filenames)
        train_thresh = int(dataset_perc['train'] * files_num)
        test_thresh = int((dataset_perc['train'] + dataset_perc['test']) * files_num)

        for cur_idx, filename in enumerate(filenames):
            filename_no_ext = '.'.join(filename.split('.')[:-2])

            # 确定数据集划分
            if cur_idx < train_thresh:
                set_type = 'train'
            elif cur_idx < test_thresh:
                set_type = 'test'
            else:
                set_type = 'val'

            output_label_path = os.path.join(output_dir, set_type, 'labels', filename_no_ext + '.txt')

            # 检查原始图像路径（支持多种格式）
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                potential_path = os.path.join(imgs_dir, label, filename_no_ext + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if img_path is None:
                print(f"Warning: Image file not found for {filename_no_ext}")
                skipped_count += 1
                continue

            # 转换图像格式
            output_img_base = os.path.join(output_dir, set_type, 'images', filename_no_ext)
            jpg_path = convert_to_jpg(img_path, output_img_base, jpg_quality)
            if jpg_path is None:
                skipped_count += 1
                continue

            # 统计文件大小
            total_original_size += os.path.getsize(img_path)
            total_converted_size += os.path.getsize(jpg_path)
            processed_count += 1

            # 处理标注文件
            try:
                img = Image.open(jpg_path)
                width, height = img.size

                with open(os.path.join(txts_dir, label, filename), 'r') as txt:
                    with open(output_label_path, 'w') as yolo_output_file:
                        txt.readline()  # ignore first line
                        line = txt.readline()

                        while line != '':
                            datas = line.strip().split()
                            if len(datas) < 5:
                                line = txt.readline()
                                continue

                            class_idx = labels.index(datas[0])
                            x0, y0, w0, h0 = int(datas[1]), int(datas[2]), int(datas[3]), int(datas[4])

                            if version == 5: # YOLOv5/v7/v8 格式 (Center X, Center Y, W, H)
                                x = (x0 + w0/2) / width
                                y = (y0 + h0/2) / height
                            elif version == 3:
                                x = x0 / width
                                y = y0 / height
                            else:
                                print("Version of YOLO error.")
                                return

                            w = w0 / width
                            h = h0 / height

                            yolo_output_file.write(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                            line = txt.readline()

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
                if os.path.exists(output_label_path):
                    os.remove(output_label_path)
                skipped_count += 1

    print(f"\n=== 转换统计 ===")
    print(f"处理图像数量: {processed_count}")
    print(f"跳过图像数量: {skipped_count}")
    print(f"原始总大小: {total_original_size/(1024 * 1024):.2f}MB")
    print(f"转换后总大小: {total_converted_size/(1024 * 1024):.2f}MB")
    if total_original_size > 0:
        print(f"压缩率: {total_converted_size/total_original_size*100:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anndir', type=str, default='./ExDark/Annnotations', help="ExDark注释文件夹路径")
    parser.add_argument('--imgdir', type=str, default='./ExDark/images', help="ExDark图像文件夹路径")
    parser.add_argument('--ratio', type=str, default='8:1:1', help="划分比率 train/test/val, default 8:1:1")
    parser.add_argument('--version', type=int, choices=[3, 5], default=5, help="转化的YOLO版本 (YOLOv7使用5)")
    parser.add_argument('--output-dir', type=str, default="./datasets/ExDark_YOLO", help="YOLO格式数据集输出的文件夹路径")
    parser.add_argument('--quality', type=int, default=95, help="JPEG质量 (1-100), 默认95")
    parser.add_argument('--seed', type=int, default=42, help="随机种子，保证划分可复现")

    args = parser.parse_args()
    ExDark2Yolo(args.anndir, args.imgdir, args.ratio, args.version, args.output_dir, args.quality, args.seed)
