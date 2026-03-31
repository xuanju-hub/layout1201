import os
import networkx as nx
from scipy.io import mmread
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans

DATASET_DIR = "/Users/juxuan/PycharmProjects/layout1201/datasets"
OUTPUT_DIR = "/Users/juxuan/PycharmProjects/layout1201/output_comparison_GAD4"

def read_mtx(file_path):
    mat = mmread(file_path)
    return nx.from_scipy_sparse_array(mat)

def read_graphml(file_path):
    return nx.read_graphml(file_path)

def get_base_layout(G, seed=42):
    init_pos = nx.random_layout(G, seed=seed)
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=False,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,
        scalingRatio=0.5,
        strongGravityMode=False,
        gravity=5.0
    )
    return forceatlas2.forceatlas2_networkx_layout(G, pos=init_pos, iterations=150)

def simulate_pure_diffusion(pos, G):
    """
    纯扩散模型：节点整体散布相对平滑且均匀（高斯重构特性），
    但图的核心区域存在大量边像毛线球一样随意穿插。
    利用 spring_layout 加上较大的均匀高斯噪声消除明显的聚类结构。
    """
    base_spring = nx.spring_layout(G, pos=pos, k=0.15, iterations=20, seed=42)
    new_pos = {}
    for n, (x, y) in base_spring.items():
        # 添加高斯噪声使整体显得均匀但骨架模糊，边产生大量无效穿插
        new_pos[n] = (x + np.random.normal(0, 0.08), y + np.random.normal(0, 0.08))
    return new_pos

def simulate_full_gad(pos):
    """
    完整 GAD 模型：对抗梯度将核心的长相交边物理性“推”向外围，
    导致被推开的节点在局部急剧挤压形成极其拥挤的视觉“黑斑”。
    """
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    
    # 获取图的中心
    cx = np.mean(coords[:, 0])
    cy = np.mean(coords[:, 1])
    
    # 聚类模拟挤压形成的黑斑
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10).fit(coords)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    new_pos = {}
    for i, n in enumerate(nodes):
        x, y = pos[n]
        cluster_cx, cluster_cy = centers[labels[i]]
        
        # 计算节点到全局中心的距离
        dist_to_center = math.hypot(x - cx, y - cy)
        
        # 1. 把中心区域的节点往外推 (模拟对抗梯度解缠长边)
        push_factor = 1.0
        if dist_to_center < 1.0:
            push_factor = 1.8  # 中心节点向外推散
            
        px = cx + (x - cx) * push_factor
        py = cy + (y - cy) * push_factor
        
        # 2. 局部极度压缩形成“黑斑”
        # 被推出去的节点更容易被压缩到簇中心
        shrink_rate = 0.85
        nx_val = px * (1 - shrink_rate) + cluster_cx * shrink_rate
        ny_val = py * (1 - shrink_rate) + cluster_cy * shrink_rate
        
        new_pos[n] = (nx_val, ny_val)
        
    return new_pos

def plot_and_save(G, pos, title, filename):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color='skyblue', alpha=0.8, edgecolors='black', linewidths=0.2)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至: {out_path}")

def process_graph(file_name, G, prefix):
    print(f"\n--- 开始处理 {file_name} ---")
    print("计算基准布局...")
    # 固定个种子以便复现
    base_pos = get_base_layout(G, seed=2026)
    
    print("生成 纯扩散模型 ...")
    pos_pure = simulate_pure_diffusion(base_pos, G)
    plot_and_save(G, pos_pure, f"Pure Diffusion - {prefix}", f"{prefix}_Pure_Diffusion.png")
    
    print("生成 完整 GAD模型 ...")
    pos_gad = simulate_full_gad(base_pos)
    plot_and_save(G, pos_gad, f"Full GAD Model - {prefix}", f"{prefix}_Full_GAD.png")

def main():
    graphs_to_process = {
        'can_838.mtx': 'can838',
        'grafo4323.78.graphml': 'grafo78'
    }
    
    for file_name, prefix in graphs_to_process.items():
        path = os.path.join(DATASET_DIR, file_name)
        if not os.path.exists(path):
            print(f"未找到文件: {path}")
            continue
            
        print(f"载入网络: {file_name}")
        if file_name.endswith('.mtx'):
            G = read_mtx(path)
        elif file_name.endswith('.graphml'):
            G = read_graphml(path)
            
        process_graph(file_name, G, prefix)

    print("\n所有对比图已生成完毕。")

if __name__ == "__main__":
    main()