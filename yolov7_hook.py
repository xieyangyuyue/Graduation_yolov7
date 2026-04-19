import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 0. 注入安全加载补丁 (防无限递归版) =================
if not hasattr(torch, '_safe_load_patched'):
    _original_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load
    torch._safe_load_patched = True
    print("✅ PyTorch 补丁首次注入成功！")
else:
    print("✅ 补丁已存在，无需重复注入。")
# ===============================================================================

from models.experimental import attempt_load

# ================= 1. 配置路径 =================
# 填入你的最佳模型权重路径 (请确认这是基线模型，还是你加了CA的改进模型)
weights_path = '/kaggle/input/models/xue0309/exdark/pytorch/default/1/best.pt'
# 挑选一张极暗的测试集图片
img_path = '/kaggle/working/Graduation_yolov7/datasets/ExDark/val/images/2015_02602.jpg'
# ===============================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("🔄 正在加载模型...")
model = attempt_load(weights_path, map_location=device)
model.eval()

# 2. 注册 Hook 提取特征图 (Hook 机制)
feature_maps = []
def hook_fn(module, input, output):
    # 将输出的 Tensor 截获并保存
    feature_maps.append(output.detach().cpu().numpy())

# 挂载 Hook 到模型的第 104 层（通常是 Neck 融合输出，送入 Head 的前一层）
# 如果报错说超出索引，或者画出来的图一片黑，可以尝试改成 101, 102, 或 105
handle = model.model[104].register_forward_hook(hook_fn)

# 3. 图像预处理
print("🖼️ 正在处理图像并提取深层特征...")
img0 = cv2.imread(img_path)
if img0 is None:
    raise FileNotFoundError(f"找不到图片：{img_path}")
img_resized = cv2.resize(img0, (640, 640))
img_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB -> CHW
img_tensor = np.ascontiguousarray(img_tensor)
img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)

# 4. 前向推理 (触发 Hook)
with torch.no_grad():
    _ = model(img_tensor)

# 5. 生成热力图
fmap = feature_maps[0][0]
# 在通道维度上求平均，得到激活强度分布
heatmap = np.mean(fmap, axis=0)
heatmap = np.maximum(heatmap, 0) # ReLU 操作，过滤负值
if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)   # 归一化到 0~1

# 6. 将热力图叠加到原图上
heatmap_resized = cv2.resize(heatmap, (img0.shape[1], img0.shape[0]))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = heatmap_color * 0.5 + img0 * 0.5

# 7. 绘图展示
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Input', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(heatmap, cmap='jet')
axes[1].set_title('Deep Feature Activation', fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
axes[2].set_title('Superimposed Heatmap', fontsize=14, fontweight='bold')
axes[2].axis('off')

output_path = '/kaggle/working/Feature_Map_Visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 特征热力图已生成并保存至：{output_path}")
plt.show()

# 释放 Hook 内存
handle.remove()
