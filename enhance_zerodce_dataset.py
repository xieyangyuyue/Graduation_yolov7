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
# 1. 定义 Zero-DCE 核心轻量级神经网络 (带高分辨率防爆显存机制)
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

    def forward(self, x_original):
        # --- 核心防爆显存机制：如果图像太大，缩小它来预测曲线 ---
        _, _, H, W = x_original.shape
        max_dim = max(H, W)
        scale_factor = 1.0
        
        # 限制网络计算的最大分辨率为 800 像素
        if max_dim > 800:
            scale_factor = 800.0 / max_dim
            # 下采样，送入网络计算曲线
            x_net = F.interpolate(x_original, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            x_net = x_original

        # CNN 计算增强曲线参数
        x1 = self.relu(self.e_conv1(x_net))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # --- 核心防爆显存机制：将预测出的曲线放大回原图尺寸 ---
        if scale_factor != 1.0:
            x_r = F.interpolate(x_r, size=(H, W), mode='bilinear', align_corners=False)

        # 在原图 (最高清分辨率) 上施加光照增强曲线计算！完美保留画质！
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x_original + r1 * (torch.pow(x_original, 2) - x_original)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image

# ==========================================
# 2. 多卡分布式数据集处理逻辑
# ==========================================
def process_yolo_dataset(input_dir, output_dir, weights_path):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if local_rank == 0:
        print(f"✅ 检测到 {world_size} 个进程。正在进行多卡任务拆分...")
    
    DCE_net = enhance_net_nopool().to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件 {weights_path}！")
        
    DCE_net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    DCE_net.eval()

    splits = ['train', 'val', 'test']
    for split in splits:
        in_split_dir = os.path.join(input_dir, split)
        if not os.path.exists(in_split_dir):
            continue
            
        out_split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(out_split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_split_dir, 'labels'), exist_ok=True)
        
        img_paths = sorted(glob(os.path.join(in_split_dir, 'images', '*.*')))
        my_img_paths = img_paths[local_rank::world_size]
        
        if local_rank == 0:
            print(f"\n🚀 开始处理 {split} 集图像 (总计 {len(img_paths)} 张)")
            
        iterator = tqdm(my_img_paths, desc=f"GPU-{local_rank} 增强中") if local_rank == 0 else my_img_paths
        
        for img_path in iterator:
            filename = os.path.basename(img_path)
            out_img_path = os.path.join(out_split_dir, 'images', filename)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img) / 255.0 
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                enhanced_image = DCE_net(img_tensor)
            
            result_img = enhanced_image.squeeze().permute(1, 2, 0).cpu().numpy()
            result_img = np.clip(result_img, 0, 1) * 255.0
            result_img = result_img.astype(np.uint8)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR) 
            
            cv2.imwrite(out_img_path, result_img)
            
            # 手动释放内存，防止显存碎片化
            del img_tensor, enhanced_image
            
        # 强制清理本轮缓存
        torch.cuda.empty_cache()
            
        if local_rank == 0:
            label_paths = glob(os.path.join(in_split_dir, 'labels', '*.txt'))
            print(f"📁 正在同步 {split} 集标签 (共 {len(label_paths)} 个)")
            for label_path in tqdm(label_paths, desc="Copying Labels"):
                filename = os.path.basename(label_path)
                shutil.copy(label_path, os.path.join(out_split_dir, 'labels', filename))

if __name__ == '__main__':
    INPUT_DIR = './datasets/ExDark'  
    OUTPUT_DIR = './datasets/ExDark_ZeroDCE'
    WEIGHTS = './Epoch99.pth'
    
    process_yolo_dataset(INPUT_DIR, OUTPUT_DIR, WEIGHTS)
    
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print("\n🎉 Zero-DCE 多卡并行增强完成！数据集已就绪！")