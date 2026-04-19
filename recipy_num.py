

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. 基础配置
# ==========================================
# 类别名称 (需与你的 ExDark.yaml 保持严格一致)
classes = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
           'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

# 你的 YOLO 格式标签存放的路径 (如果使用了 ZeroDCE 增强的数据集，请修改为对应的绝对路径)
# 这里涵盖了 train, val, test 三个文件夹
label_dirs = [
    'datasets/ExDark/train/labels',
    'datasets/ExDark/val/labels',
    'datasets/ExDark/test/labels'
]

# 用于存放统计结果的容器
class_counts = {c: 0 for c in classes}
widths = []
heights = []

print("⏳ 正在读取并分析标签文件，请稍候...")

# ==========================================
# 2. 读取 YOLO 格式的 TXT 标签
# ==========================================
processed_files = 0
for d in label_dirs:
    if not os.path.exists(d):
        continue
    for txt_file in glob.glob(os.path.join(d, '*.txt')):
        processed_files += 1
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_idx = int(parts[0])  # 类别索引
                    w = float(parts[3])    # 归一化后的宽度
                    h = float(parts[4])    # 归一化后的高度
                    
                    if c_idx < len(classes):
                        class_counts[classes[c_idx]] += 1
                    
                    widths.append(w)
                    heights.append(h)

print(f"✅ 共解析了 {processed_files} 个标签文件，包含 {len(widths)} 个目标框。")

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签(若环境支持)
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 3. 绘制类别数量柱状图
# ==========================================
plt.figure(figsize=(12, 6))
# 转换为 DataFrame 方便使用 seaborn
df_counts = pd.DataFrame({
    'Category': list(class_counts.keys()),
    'Count': list(class_counts.values())
})
# 按数量降序排列，让图表更美观
df_counts = df_counts.sort_values(by='Count', ascending=False)

ax = sns.barplot(x='Category', y='Count', data=df_counts, palette='viridis')
plt.title('ExDark', fontsize=16, pad=15)# 数据集各类别目标数量分布
plt.xlabel('Target category', fontsize=12)
plt.ylabel('Instances', fontsize=12)
plt.xticks(rotation=45, fontsize=11)

# 在柱子上添加具体数字
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom', 
                fontsize=10, color='black', xytext=(0, 4), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig('ExDark_Class_Distribution.png', dpi=300)
plt.show()
print("📸 类别分布柱状图已保存为: ExDark_Class_Distribution.png")

# ==========================================
# 4. 绘制边界框长宽比 (宽高) 散点图
# ==========================================
plt.figure(figsize=(8, 8))
# 使用半透明度 (alpha=0.2) 展现数据密集程度
sns.scatterplot(x=widths, y=heights, alpha=0.2, s=15, color='#e74c3c', edgecolor=None)

# 绘制参考线（例如 1:1, 1:2, 2:1 的长宽比辅助线）
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='1:1 ')
plt.plot([0, 1], [0, 2], 'b--', alpha=0.5, label='1:2 ')
plt.plot([0, 1], [0, 0.5], 'g--', alpha=0.5, label='2:1')

plt.title('ExDark Normalized width-height scatter plot of target bounding box', fontsize=16, pad=15)
plt.xlabel('Normalized Width', fontsize=12)
plt.ylabel('Normalized Height', fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('ExDark_BBox_Scatter.png', dpi=300)
plt.show()
print("📸 长宽比散点图已保存为: ExDark_BBox_Scatter.png")
