import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm

# ==========================================
# 1. 定义 Zero-DCE 核心轻量级神经网络
# ==========================================
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image

# ==========================================
# 2. 数据集批量处理逻辑 (保持YOLO格式对齐)
# ==========================================
def process_yolo_dataset(input_dir, output_dir, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用 {device} 加载 Zero-DCE 模型...")
    
    # 实例化并加载预训练权重
    DCE_net = enhance_net_nopool().to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件 {weights_path}！请先执行 wget 下载。")
    DCE_net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    DCE_net.eval() # 开启评估模式

    splits = ['train', 'val', 'test']
    for split in splits:
        in_split_dir = os.path.join(input_dir, split)
        if not os.path.exists(in_split_dir):
            continue
            
        out_split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(out_split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_split_dir, 'labels'), exist_ok=True)
        
        # 处理图像
        img_paths = glob(os.path.join(in_split_dir, 'images', '*.*'))
        print(f"\n--- 正在使用 Zero-DCE 增强 {split} 集图像 (共 {len(img_paths)} 张) ---")
        
        for img_path in tqdm(img_paths, desc="Deep Enhancing"):
            filename = os.path.basename(img_path)
            out_img_path = os.path.join(out_split_dir, 'images', filename)
            
            # 读取图片并转换为 Tensor
            img = cv2.imread(img_path)
            if img is None: continue
            
            # cv2是BGR，Zero-DCE最好使用RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img) / 255.0 # 归一化到 [0, 1]
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 使用深度学习提亮
            with torch.no_grad():
                enhanced_image = DCE_net(img)
            
            # 将 Tensor 转回 numpy 图像
            result_img = enhanced_image.squeeze().permute(1, 2, 0).cpu().numpy()
            result_img = np.clip(result_img, 0, 1) * 255.0
            result_img = result_img.astype(np.uint8)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR) # 转回BGR保存
            
            cv2.imwrite(out_img_path, result_img)
            
        # 同步拷贝标签
        label_paths = glob(os.path.join(in_split_dir, 'labels', '*.txt'))
        print(f"--- 正在同步 {split} 集标签 (共 {len(label_paths)} 个) ---")
        for label_path in tqdm(label_paths, desc="Copying Labels"):
            filename = os.path.basename(label_path)
            shutil.copy(label_path, os.path.join(out_split_dir, 'labels', filename))

if __name__ == '__main__':
    # 配置路径
    INPUT_DIR = './datasets/ExDark'  
    OUTPUT_DIR = './datasets/ExDark_ZeroDCE'
    WEIGHTS = './Epoch99.pth' # 刚才下载的权重文件
    
    process_yolo_dataset(INPUT_DIR, OUTPUT_DIR, WEIGHTS)
    print("\n✅ Zero-DCE 深度学习数据集增强完成！")