import os
import networkx as nx
from scipy.io import mmread
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import hashlib

DATASET_DIR = "/Users/juxuan/PycharmProjects/layout1201/datasets"
OUTPUT_DIR = "/Users/juxuan/PycharmProjects/layout1201/output_images"

def read_mtx(file_path):
    mat = mmread(file_path)
    return nx.from_scipy_sparse_array(mat)

def read_graphml(file_path):
    return nx.read_graphml(file_path)

def apply_fa2(G, seed=42):
    print(f"Applying ForceAtlas2 to graph with {G.number_of_nodes()} nodes...")

    # 固定随机种子（保证每次初始状态一致）
    random.seed(seed)
    np.random.seed(seed)

    # 不再用 pos=None，改为固定seed的初始布局
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

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=init_pos, iterations=100)
    return positions

def stable_seed_from_name(name: str, base: int = 2026) -> int:
    """
    根据文件名生成稳定随机种子：
    - 同名文件 seed 恒定
    - 不同文件 seed 不同
    """
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base) % (2**31 - 1)

def calculate_node_overlaps(G, pos, node_radius):
    """
    计算节点重叠对的数量
    :param G: NetworkX 图对象
    :param pos: 布局坐标字典 {node: (x, y)}
    :param node_radius: 节点在坐标系中的实际半径
    :return: 重叠的节点对数量
    """
    nodes = sorted(G.nodes(), key=str)
    overlap_count = 0
    min_dist = node_radius * 2
    
    # 两两对比，时间复杂度 O(N^2)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            # 计算欧氏距离
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < min_dist:
                overlap_count += 1
                
    return overlap_count

def calculate_edge_crossings(G, pos):
    """
    计算边交叉的总数量
    :param G: NetworkX 图对象
    :param pos: 布局坐标字典 {node: (x, y)}
    :return: 交叉边的对数
    """
    edges = sorted(G.edges(), key=lambda e: (str(e[0]), str(e[1])))
    crossing_count = 0
    
    # 两两对比，时间复杂度 O(E^2)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            # 如果两条边共享同一个顶点，跳过判定（不算作交叉）
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            A, B = pos[u1], pos[v1]
            C, D = pos[u2], pos[v2]
            
            if edges_intersect(A, B, C, D):
                crossing_count += 1
                
    return crossing_count

def enforce_overlaps(G, pos, target_overlaps, node_radius):
    """
    根据目标重叠数微调坐标：
    - 重叠需要增加时：拉近最近的非重叠节点对；
    - 重叠需要减少时：推开当前重叠节点对；
    位移幅度与“重叠差值”成正相关，差值越大变化越明显。
    """
    nodes = sorted(G.nodes(), key=str)  # 固定顺序，保证可复现
    min_dist = node_radius * 2

    def collect_pairs():
        overlap_pairs = []
        non_overlap_pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                d = math.hypot(x2 - x1, y2 - y1)
                if d < min_dist:
                    overlap_pairs.append((d, u, v))
                else:
                    non_overlap_pairs.append((d, u, v))
        overlap_pairs.sort(key=lambda x: x[0])      # 重叠最严重在前
        non_overlap_pairs.sort(key=lambda x: x[0])  # 最接近重叠在前
        return overlap_pairs, non_overlap_pairs

    overlap_pairs, non_overlap_pairs = collect_pairs()
    current = len(overlap_pairs)
    print(f"  [微调] 当前重叠数: {current}, 目标重叠数: {target_overlaps}")

    if current == target_overlaps:
        return pos

    # 差值比例：越大代表要改动越多 -> 位移越大
    diff = abs(target_overlaps - current)
    base = max(1, target_overlaps if target_overlaps > 0 else current)
    diff_ratio = min(1.0, diff / base)

    # 增加重叠：将最近的非重叠节点对拉近，diff_ratio 越大，拉得越狠
    if current < target_overlaps:
        need = target_overlaps - current
        changed = 0
        for d, u, v in non_overlap_pairs:
            if changed >= need:
                break
            ux, uy = pos[u]
            vx, vy = pos[v]
            dx, dy = ux - vx, uy - vy
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue

            # 目标距离：越高重叠需求 -> 目标距离越小（更重叠）
            target_d = min_dist * (0.95 - 0.45 * diff_ratio)
            target_d = max(0.05 * min_dist, target_d)

            if dist > target_d:
                move = dist - target_d
                nx_, ny_ = dx / dist, dy / dist
                pos[v] = (vx + nx_ * move, vy + ny_ * move)
                changed += 1

    # 减少重叠：将重叠节点推开，diff_ratio 越大，推得越远
    else:
        need = current - target_overlaps
        changed = 0
        for d, u, v in overlap_pairs:
            if changed >= need:
                break
            ux, uy = pos[u]
            vx, vy = pos[v]
            dx, dy = vx - ux, vy - uy
            dist = math.hypot(dx, dy)

            if dist == 0:
                # 完全重合给一个固定方向，避免随机性
                dx, dy = 1.0, 0.0
                dist = 1.0

            nx_, ny_ = dx / dist, dy / dist

            # 目标距离：差值越大，推得越远
            target_d = min_dist * (1.01 + 1.8 * diff_ratio)
            pos[v] = (ux + nx_ * target_d, uy + ny_ * target_d)
            changed += 1

    return pos

def edges_intersect(A, B, C, D):
    """判断线段AB与线段CD是否相交"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    """判断三点是否为逆时针方向"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def plot_graph(G, pos, title, with_labels=False):
    plt.figure(figsize=(10, 10))
    plt.title(f"ForceAtlas2 Layout - {title}")
    
    # 隐藏坐标轴
    plt.axis('off')
    
    # 绘制网络图
    #node_size = 10 if G.number_of_nodes() > 1000 else 300 if with_labels else 30
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    # 画节点编号
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, f"{title}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至: {output_path}")

def apply_8dir_radial_warp(pos, dir_strengths, power=2.0):
    """
    8方向径向缩放（独立控制，每个方向可放大/缩小）
    参数:
      pos: {node: (x, y)}
      dir_strengths: 长度为8的列表，顺序为
        [E, NE, N, NW, W, SW, S, SE]
        例如: [0.2, 0.1, 0.0, -0.1, -0.2, 0.0, 0.1, 0.2]
      power: 外围增强指数，越大越强调外围节点变化
    说明:
      strength > 0 -> 径向放大
      strength < 0 -> 径向缩小
      最终缩放: scale = 1 + strength(theta) * (r_norm ** power)
    """
    if not pos:
        return pos
    if len(dir_strengths) != 8:
        raise ValueError("dir_strengths 必须是8个方向参数: [E, NE, N, NW, W, SW, S, SE]")

    nodes = list(pos.keys())
    cx = sum(pos[n][0] for n in nodes) / len(nodes)
    cy = sum(pos[n][1] for n in nodes) / len(nodes)

    # 最大半径
    r_max = 0.0
    for n in nodes:
        x, y = pos[n]
        r = math.hypot(x - cx, y - cy)
        r_max = max(r_max, r)
    if r_max == 0:
        return pos

    # 8方向角度步长 45°
    step = math.pi / 4.0

    def strength_by_angle(theta):
        # theta 映射到 [0, 2pi)
        t = theta % (2.0 * math.pi)
        idx = int(t // step)              # 当前扇区
        nxt = (idx + 1) % 8               # 下一个扇区
        local = (t - idx * step) / step   # [0,1] 线性插值
        return dir_strengths[idx] * (1 - local) + dir_strengths[nxt] * local

    new_pos = {}
    for n in nodes:
        x, y = pos[n]
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy)
        if r == 0:
            new_pos[n] = (x, y)
            continue

        theta = math.atan2(dy, dx)  # 0=E, pi/2=N, pi=W, -pi/2=S
        s = strength_by_angle(theta)
        r_norm = r / r_max
        scale = 1.0 + s * (r_norm ** power)

        # 防止缩小过度翻转
        if scale < 0.05:
            scale = 0.05

        new_pos[n] = (cx + dx * scale, cy + dy * scale)

    return new_pos
#[E, NE, N, NW, W, SW, S, SE]
# dir8_map = {
#     'plskz362.mtx':            [0.20, 0.10, -0.20, 1.00, 0.10, -0.20, 0.10, 0.15],
#     'can_838.mtx':             [0.15, 0.05, -0.05, 3.60, -0.05, -0.30, 0.10, -0.20],
#     'grafo4323.78.graphml':    [0.20, 0.10, -0.30, -0.25, -0.15, -0.15, 0.00, 0.20],
#     'grafo9873.35.graphml':    [0.10, 0.15, -0.50, -0.10, -0.20, -0.10, 0.05, -0.50],
# }

dir8_map = {
    'plskz362.mtx':            [0.0, 1.00, 0.0, 3.00, 1.00, 0.0, 1.80, 0.80],
    'can_838.mtx':             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'grafo4323.78.graphml':    [0.00, 0.0, 0.00, 1.60, 1.00, 0.00, -0.50, 0.00],
    'grafo9873.35.graphml':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

def main():
    graphs = [
        #'plskz362.mtx',
        'can_838.mtx'
        #'grafo4323.78.graphml'
        #'grafo9873.35.graphml'
    ]
    
    # 为特定图设定目标重叠数量
    target_overlaps_map = {
        'grafo4323.78.graphml': 16,  #
        'grafo9873.35.graphml': 4,  #
        'plskz362.mtx': 16,         #
        'can_838.mtx':  30          #
    }

    print("--- 处理所有图 (ForceAtlas2) 并绘制图像 ---")
    for file_name in graphs:
        path = os.path.join(DATASET_DIR, file_name)
        if os.path.exists(path):
            if file_name.endswith('.mtx'):
                G = read_mtx(path)
            elif file_name.endswith('.graphml'):
                G = read_graphml(path)
            else:
                continue
                
            # 为每个文件生成“稳定但不同”的 seed
            seed = stable_seed_from_name(file_name)
            print(f"[{file_name}] 使用固定 seed: {seed}")

            # 计算布局
            layout = apply_fa2(G, seed=seed)

            node_rad = 0.5
            has_target_overlaps = file_name in target_overlaps_map

            # 先做重叠微调
            if has_target_overlaps:
                target = target_overlaps_map[file_name]
                print(f"[{file_name}] 开始调整重叠数至目标: {target}")
                layout = enforce_overlaps(G, layout, target_overlaps=target, node_radius=node_rad)

            # 再做 8 方向径向缩放（这里改 dir8_map 才会体现在最终图片）
            dir_strengths = dir8_map.get(file_name)
            if dir_strengths is not None:
                layout = apply_8dir_radial_warp(layout, dir_strengths=dir_strengths, power=2.0)

            # 统计放在最终布局之后
            overlaps = calculate_node_overlaps(G, layout, node_radius=node_rad)
            crossings = calculate_edge_crossings(G, layout)
            print(f"{file_name} 最终节点重叠数: {overlaps}, 边交叉数: {crossings}")

            # 最后保存图
            plot_graph(G, layout, file_name, with_labels=False)
            
            print(f"已完成 {file_name} 的处理。\n")
        else:
            print(f"未找到文件: {path}\n")

if __name__ == "__main__":
    main()
