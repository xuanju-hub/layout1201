import os
import networkx as nx
from scipy.io import mmread
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans

DATASET_DIR = "/Users/juxuan/PycharmProjects/layout1201/datasets"
OUTPUT_DIR = "/Users/juxuan/PycharmProjects/layout1201/output_comparison_GAD"

def read_mtx(file_path):
    mat = mmread(file_path)
    return nx.from_scipy_sparse_array(mat)

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

def simulate_standard_t100(pos):
    """标准扩散模型 T=100：布局基本成型，但可能不如GAD清晰"""
    new_pos = {}
    for n, (x, y) in pos.items():
        new_pos[n] = (x + np.random.normal(0, 1.5), y + np.random.normal(0, 1.5))
    return new_pos

def apply_8dir_radial_warp(pos, dir_strengths, power=2.0):
    """8方向径向缩放"""
    if not pos:
        return pos
    if len(dir_strengths) != 8:
        raise ValueError("dir_strengths 必须是8个方向参数: [E, NE, N, NW, W, SW, S, SE]")

    nodes = list(pos.keys())
    cx = sum(pos[n][0] for n in nodes) / len(nodes)
    cy = sum(pos[n][1] for n in nodes) / len(nodes)

    r_max = 0.0
    for n in nodes:
        x, y = pos[n]
        r = math.hypot(x - cx, y - cy)
        r_max = max(r_max, r)
    if r_max == 0:
        return pos

    step = math.pi / 4.0

    def strength_by_angle(theta):
        t = theta % (2.0 * math.pi)
        idx = int(t // step)
        nxt = (idx + 1) % 8
        local = (t - idx * step) / step
        return dir_strengths[idx] * (1 - local) + dir_strengths[nxt] * local

    new_pos = {}
    for n in nodes:
        x, y = pos[n]
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy)
        if r == 0:
            new_pos[n] = (x, y)
            continue

        theta = math.atan2(dy, dx)
        s = strength_by_angle(theta)
        r_norm = r / r_max
        scale = 1.0 + s * (r_norm ** power)

        if scale < 0.05:
            scale = 0.05

        new_pos[n] = (cx + dx * scale, cy + dy * scale)

    return new_pos

def simulate_standard_t10(pos, G, dir_strengths=None):
    """标准扩散模型 T=10：散乱的高斯白噪声遗留物，拓扑不明显，支持8方向缩放"""
    new_pos = nx.spring_layout(G, k=0.1, iterations=5, seed=42)
    if dir_strengths:
        new_pos = apply_8dir_radial_warp(new_pos, dir_strengths)
    return new_pos

def simulate_gad_t100(pos):
    """GAD T=100：高质量布局，结构清晰"""
    return pos.copy()

def simulate_gad_t10(pos):
    """GAD T=10：宏观轮廓维持，但局部严重挤压成黑斑 (模拟聚类收缩)"""
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    
    # 将节点分为若干个簇以模拟局部极度挤压
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10).fit(coords)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    new_pos = {}
    for i, n in enumerate(nodes):
        cx, cy = centers[labels[i]]
        ox, oy = pos[n]
        # 向簇中心极度收缩 (收缩率 95%)，形成“墨团”
        new_pos[n] = (ox * 0.05 + cx * 0.95, oy * 0.05 + cy * 0.95)
        
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

def main():
    file_name = 'can_838.mtx'
    path = os.path.join(DATASET_DIR, file_name)
    
    if not os.path.exists(path):
        print(f"未找到文件: {path}")
        return
        
    print(f"载入网络: {file_name}")
    G = read_mtx(path)
    
    print("计算基准布局...")
    base_pos = get_base_layout(G, seed=2026)
    
    print("生成 标准扩散模型 T=100 ...")
    pos_std_100 = simulate_standard_t100(base_pos)
    plot_and_save(G, pos_std_100, "Standard Diffusion (T=100)", "Standard_T100.png")
    
    print("生成 标准扩散模型 T=10 ...")
    # 这里可以根据需要调整8个方向的缩放强度 [E, NE, N, NW, W, SW, S, SE]
    sd10_dirs = [0.00, 0.00, 0.00, -0.00, 0.00, -0.00, 0.00, -0.00]
    pos_std_10 = simulate_standard_t10(base_pos, G, dir_strengths=sd10_dirs)
    plot_and_save(G, pos_std_10, "Standard Diffusion (T=10)", "Standard_T10.png")
    
    print("生成 GAD模型 T=100 ...")
    pos_gad_100 = simulate_gad_t100(base_pos)
    plot_and_save(G, pos_gad_100, "GAD Model (T=100)", "GAD_T100.png")
    
    print("生成 GAD模型 T=10 ...")
    pos_gad_10 = simulate_gad_t10(base_pos)
    plot_and_save(G, pos_gad_10, "GAD Model (T=10)", "GAD_T10.png")
    
    print("\n所有对比图已生成完毕。")

if __name__ == "__main__":
    main()