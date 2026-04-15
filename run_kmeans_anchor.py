import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 添加随机数种子 ==========
# 设置随机种子确保结果可重现
# 使用42作为种子（深度学习领域常用值）
np.random.seed(42)
# ==================================

def cascade_iou(boxes, clusters):
    # 计算所有边界框与聚类中心的 IoU
    box_area = boxes[:, 0] * boxes[:, 1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    inter_w = np.minimum(boxes[:, 0][:, np.newaxis], clusters[:, 0])
    inter_h = np.minimum(boxes[:, 1][:, np.newaxis], clusters[:, 1])
    inter_area = inter_w * inter_h

    iou = inter_area / (box_area[:, np.newaxis] + cluster_area - inter_area + 1e-16)
    return iou

def kmeans_plus_plus(boxes, k=9, max_iter=100):
    n_boxes = boxes.shape[0]
    clusters = np.empty((k, 2))

    # 1. K-Means++ 初始化策略
    clusters[0] = boxes[np.random.choice(n_boxes)]
    for i in range(1, k):
        iou = cascade_iou(boxes, clusters[:i])
        distances = 1 - np.max(iou, axis=1)
        probs = distances ** 2 / np.sum(distances ** 2)
        clusters[i] = boxes[np.random.choice(n_boxes, p=probs)]

    # 2. 常规 K-Means 迭代更新
    for epoch in range(max_iter):
        iou = cascade_iou(boxes, clusters)
        nearest_clusters = np.argmax(iou, axis=1)

        new_clusters = np.empty_like(clusters)
        for i in range(k):
            # 获取属于第 i 个簇的所有框
            cluster_boxes = boxes[nearest_clusters == i]
            if len(cluster_boxes) > 0:
                # 使用中位数更新簇心（比平均值对异常值更鲁棒）
                new_clusters[i] = np.median(cluster_boxes, axis=0)
            else:
                new_clusters[i] = clusters[i]

        # 检查是否收敛
        if np.all(clusters == new_clusters):
            print(f"聚类在第 {epoch} 轮收敛。")
            break
        clusters = new_clusters

    return clusters, nearest_clusters

if __name__ == '__main__':
    # 1. 设置你的训练集标签路径 (请确保路径正确)
    label_path = 'datasets/ExDark/train/labels/*.txt'
    img_size = 640 # YOLOv7 训练的图像尺寸

    # 2. 读取所有真实框的 w 和 h
    boxes = []
    print("正在读取标签文件...")
    for file in tqdm(glob.glob(label_path)):
        with open(file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    # 将归一化的比例还原为 640x640 下的真实像素尺寸
                    boxes.append([w * img_size, h * img_size])

    boxes = np.array(boxes)
    print(f"共提取到 {len(boxes)} 个真实边界框。")

    # 3. 运行 K-Means++
    print("开始 K-Means++ 聚类...")
    k = 9 # YOLOv7 需要 9 个 Anchor (P3, P4, P5 各 3 个)
    clusters, nearest_clusters = kmeans_plus_plus(boxes, k=k)

    # 4. 按面积从小到大排序
    area = clusters[:, 0] * clusters[:, 1]
    clusters = clusters[np.argsort(area)]

    # 5. 格式化输出为 YOLOv7 需要的格式
    print("\n================ 计算完成 ================")
    print("请将以下 9 组 Anchor 替换到 yolov7-CA-Neck.yaml 的 anchors 字段中：\n")

    # 四舍五入取整
    anchors_int = np.round(clusters).astype(int)
    print("anchors:")
    print(f"  - [{anchors_int[0][0]},{anchors_int[0][1]}, {anchors_int[1][0]},{anchors_int[1][1]}, {anchors_int[2][0]},{anchors_int[2][1]}]  # P3/8")
    print(f"  - [{anchors_int[3][0]},{anchors_int[3][1]}, {anchors_int[4][0]},{anchors_int[4][1]}, {anchors_int[5][0]},{anchors_int[5][1]}]  # P4/16")
    print(f"  - [{anchors_int[6][0]},{anchors_int[6][1]}, {anchors_int[7][0]},{anchors_int[7][1]}, {anchors_int[8][0]},{anchors_int[8][1]}]  # P5/32")

    # 6. 生成可视化图表 (用于毕业论文)
    plt.figure(figsize=(10, 8))
    plt.scatter(boxes[:, 0], boxes[:, 1], c=nearest_clusters, cmap='viridis', s=2, alpha=0.5, label='Ground Truth')
    plt.scatter(clusters[:, 0], clusters[:, 1], c='red', marker='X', s=150, edgecolors='black', label='Anchor Centers')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('ExDark Dataset Anchor Clustering (K-Means++)')
    plt.legend()
    plt.savefig('anchor_clusters.png', dpi=300)
    print("\n已生成聚类散点图 anchor_clusters.png，可用于毕业论文！")
