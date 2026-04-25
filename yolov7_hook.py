import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

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
# ===================================================================

from models.experimental import attempt_load


class MultiFeatureExtractor:
    def __init__(self, model, layer_indices):
        self.features = {}
        self.hooks = []
        self.layer_indices = layer_indices
        for idx in layer_indices:
            hook = model.model[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, idx):
        def hook_fn(module, inputs, output):
            self.features[idx] = output.detach().clone()

        return hook_fn

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def build_heatmap(feature_tensor):
    fmap = feature_tensor[0].detach().cpu().numpy()
    heatmap = np.mean(fmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    return heatmap


def overlay_heatmap(image_bgr, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = heatmap_color * 0.5 + image_bgr * 0.5
    return heatmap_resized, np.uint8(superimposed)


# ================= 1. 配置路径 =================
weights_path = '/kaggle/input/models/xue0309/exdark/pytorch/default/1/best.pt'
img_path = '/kaggle/working/Graduation_yolov7/datasets/ExDark/val/images/2015_02602.jpg'
output_dir = Path('/kaggle/working/hook_vis_outputs')
layer_map = {
    102: 'P3_small',
    104: 'P4_medium',
    106: 'P5_large',
}
# ==============================================

output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("🔄 正在加载模型...")
model = attempt_load(weights_path, map_location=device)
model.eval()

extractor = MultiFeatureExtractor(model, list(layer_map.keys()))

print("🖼️ 正在处理图像并提取多尺度特征...")
img0 = cv2.imread(img_path)
if img0 is None:
    raise FileNotFoundError(f"找不到图片：{img_path}")

img_resized = cv2.resize(img0, (640, 640))
img_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1)
img_tensor = np.ascontiguousarray(img_tensor)
img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)

with torch.no_grad():
    _ = model(img_tensor)

for idx, layer_name in layer_map.items():
    if idx not in extractor.features:
        print(f"⚠️ 第 {idx} 层未捕获到特征图，跳过。")
        continue

    heatmap = build_heatmap(extractor.features[idx])
    heatmap_resized, superimposed_img = overlay_heatmap(img0, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title(f'{layer_name} Activation', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'{layer_name} Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    save_path = output_dir / f'{layer_name}_feature_map.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {layer_name} 热力图已保存至：{save_path}")

extractor.close()
print(f"✅ 全部 Hook 可视化完成，输出目录：{os.fspath(output_dir)}")
