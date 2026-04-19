import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 配置路径 =================
# 请确保这是你训练集 labels 的文件夹路径（YOLO格式的 txt 文件）
label_dir = 'datasets/ExDark/train/labels/' 
# ===============================================

classes = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

# 2. 读取并解析所有 YOLO 标签
data = []
txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
print(f"🔄 正在解析 {len(txt_files)} 个标签文件，请稍候...")

for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id, x_c, y_c, w, h = map(float, parts)
                data.append([classes[int(cls_id)], x_c, y_c, w, h])

df = pd.DataFrame(data, columns=['Class', 'X_center', 'Y_center', 'Width', 'Height'])

# 3. 开始绘制高逼格学术三联图
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.set_theme(style="whitegrid")

# 图 (a): 类别分布柱状图
sns.countplot(data=df, y='Class', order=df['Class'].value_counts().index, 
              palette='viridis', ax=axes[0])
axes[0].set_title('(a) Class Distribution', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Number of Instances', fontsize=12)
axes[0].set_ylabel('')

# 图 (b): 目标框尺寸分布散点图
# 散点图透明度设为 0.1，可以清晰看出点密集的区域（小目标区域）
axes[1].scatter(df['Width'], df['Height'], alpha=0.1, color='teal', s=10)
axes[1].set_title('(b) Bounding Box Size Distribution', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Normalized Width', fontsize=12)
axes[1].set_ylabel('Normalized Height', fontsize=12)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

# 图 (c): 目标中心点空间热力图
sns.kdeplot(x=df['X_center'], y=df['Y_center'], cmap="mako", fill=True, 
            thresh=0, levels=100, ax=axes[2])
axes[2].set_title('(c) Object Center Spatial Heatmap', fontsize=16, fontweight='bold')
axes[2].set_xlabel('Normalized X Center', fontsize=12)
axes[2].set_ylabel('Normalized Y Center', fontsize=12)
axes[2].set_xlim(0, 1)
axes[2].set_ylim(1, 0) # 图像Y轴通常是向下增长的

plt.tight_layout()
output_path = 'ExDark_EDA_Analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ EDA 学术分析图已生成并保存至：{output_path}")
plt.show()
