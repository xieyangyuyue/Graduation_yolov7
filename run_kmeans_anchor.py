import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 1. 跨平台全局随机数种子锁定 ==========
def seed_everything(seed=42):
    # 锁定 Python 内置随机数生成器
    random.seed(seed)
    # 锁定 Numpy 随机数生成器
    np.random.seed(seed)
    # 锁定 Hash 种子，保证字典/集合遍历顺序一致
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔒 已锁定全局随机种子 (Seed={seed})，确保跨平台结果绝对一致。")

seed_everything(42)
# ===============================================

def cascade_iou(boxes, clusters):
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

    clusters[0] = boxes[np.random.choice(n_boxes)]
    for i in range(1, k):
        iou = cascade_iou(boxes, clusters[:i])
        distances = 1 - np.max(iou, axis=1)
        probs = distances ** 2 / np.sum(distances ** 2)
        clusters[i] = boxes[np.random.choice(n_boxes, p=probs)]

    for epoch in range(max_iter):
        iou = cascade_iou(boxes, clusters)
        nearest_clusters = np.argmax(iou, axis=1)

        new_clusters = np.empty_like(clusters)
        for i in range(k):
            cluster_boxes = boxes[nearest_clusters == i]
            if len(cluster_boxes) > 0:
                new_clusters[i] = np.median(cluster_boxes, axis=0)
            else:
                new_clusters[i] = clusters[i]

        if np.all(clusters == new_clusters):
            print(f"✨ K-Means++ 在第 {epoch} 轮成功收敛。")
            break
        clusters = new_clusters

    return clusters, nearest_clusters

def genetic_algorithm_evolution(boxes, clusters, epochs=100000, mutation_prob=0.9):
    print(f"🚀 启动遗传算法微调 ({epochs} 轮)...")
    best_clusters = clusters.copy()
    
    def fitness(c):
        iou = cascade_iou(boxes, c)
        return np.mean(np.max(iou, axis=1)) # 平均最大 IoU
    
    best_fitness = fitness(best_clusters)
    print(f"初始适应度 (K-Means++ 结果): {best_fitness:.4f}")
    
    for _ in tqdm(range(epochs), desc="GA 进化中"):
        # 变异操作：以高斯分布进行微小扰动
        mutation = np.ones_like(clusters)
        if np.random.random() < mutation_prob:
            mutation = np.exp(np.random.randn(*clusters.shape) * 0.1)
        
        mutated_clusters = best_clusters * mutation
        mutated_fitness = fitness(mutated_clusters)
        
        # 优胜劣汰
        if mutated_fitness > best_fitness:
            best_fitness = mutated_fitness
            best_clusters = mutated_clusters
            
    print(f"✅ 遗传算法进化完成！最终适应度: {best_fitness:.4f}")
    return best_clusters

if __name__ == '__main__':
    # 1. 设置路径 (由于使用了 ZeroDCE 增强数据，路径需对齐)
    label_path = 'datasets/ExDark/train/labels/*.txt'
    img_size = 640 

    boxes = []
    print("📂 正在读取标签文件...")
    for file in tqdm(glob.glob(label_path)):
        with open(file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    boxes.append([w * img_size, h * img_size])

    boxes = np.array(boxes)
    print(f"📊 共提取到 {len(boxes)} 个真实边界框。")

    # 2. 运行 K-Means++
    print("\n" + "="*40)
    k = 9 
    clusters, _ = kmeans_plus_plus(boxes, k=k)

    # 3. [修复点] 真正调用遗传算法进行微调！
    print("\n" + "="*40)
    clusters = genetic_algorithm_evolution(boxes, clusters, epochs=100000)

    # 4. 按面积排序并格式化输出
    area = clusters[:, 0] * clusters[:, 1]
    clusters = clusters[np.argsort(area)]

    print("\n================ 计算完成 ================")
    print("👇 请将以下 9 组 Anchor 复制并完全替换 yolov7-CA-Neck.yaml 中的 anchors 字段 👇\n")

    anchors_int = np.round(clusters).astype(int)
    print("anchors:")
    print(f"  - [{anchors_int[0][0]},{anchors_int[0][1]}, {anchors_int[1][0]},{anchors_int[1][1]}, {anchors_int[2][0]},{anchors_int[2][1]}]  # P3/8 (小目标)")
    print(f"  - [{anchors_int[3][0]},{anchors_int[3][1]}, {anchors_int[4][0]},{anchors_int[4][1]}, {anchors_int[5][0]},{anchors_int[5][1]}]  # P4/16 (中目标)")
    print(f"  - [{anchors_int[6][0]},{anchors_int[6][1]}, {anchors_int[7][0]},{anchors_int[7][1]}, {anchors_int[8][0]},{anchors_int[8][1]}]  # P5/32 (大目标)")
    print("\n⚠️ 记得在下次训练命令中加上 --noautoanchor 参数！")

    # 5. 重新计算最近类并生成图表
    _, nearest_clusters = kmeans_plus_plus(boxes, k=k, max_iter=1)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(boxes[:, 0], boxes[:, 1], c=nearest_clusters, cmap='viridis', s=2, alpha=0.5, label='Ground Truth')
    plt.scatter(clusters[:, 0], clusters[:, 1], c='red', marker='X', s=150, edgecolors='black', label='Final Anchors (K-Means++ & GA)')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('ExDark Dataset Optimized Anchor Clustering')
    plt.legend()
    plt.savefig('optimized_anchor_clusters.png', dpi=300)
    print("\n📸 已生成聚类散点图 optimized_anchor_clusters.png，可用于毕业论文！")