import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread
import warnings
import time
import numpy as np

# 忽略警告
warnings.filterwarnings("ignore")

# 尝试导入社区检测库（如果没装，会自动降级处理）
try:
    import community.community_louvain as community_louvain

    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("提示: 未安装 python-louvain，将使用简单的颜色映射。")

# 数据集目录 (对应你截图中的文件夹名)
DATA_DIR = "datasets"


def simulate_deep_learning_layout(G, model_type='GNN', epochs=100):
    """
    模拟深度学习模型的布局算法
    model_type: 'GNN', 'VAE', 'Diffusion'
    """
    np.random.seed(42)  # 确保可重复性

    n_nodes = G.number_of_nodes()

    if model_type == 'GNN':
        # 模拟图神经网络(GNN)的输出
        # GNN通常产生层次化、社区化的布局
        pos = nx.spring_layout(G, k=0.1, iterations=100, seed=42)
        # 添加一些"神经网络"特征：轻微的聚类效果
        for _ in range(20):  # 微调迭代
            pos = _gnn_style_adjustment(G, pos)

    elif model_type == 'VAE':
        # 模拟变分自编码器(VAE)的输出
        # VAE通常产生更分散、有规律的布局
        pos = nx.kamada_kawai_layout(G)
        # 添加VAE特征：圆形/径向分布趋势
        center = np.mean(list(pos.values()), axis=0)
        for node in pos:
            vec = pos[node] - center
            # 添加径向偏移
            pos[node] = center + vec * 1.2

    elif model_type == 'Diffusion':
        # 模拟扩散模型(Diffusion)的输出
        # 扩散模型产生平滑、连续的布局
        pos = nx.spring_layout(G, k=0.2, iterations=200, seed=42)
        # 添加扩散特征：更平滑的节点分布
        pos = _diffusion_style_smoothing(G, pos)

    return pos

def _gnn_style_adjustment(G, pos):
    """GNN风格的微调：增强社区结构"""
    new_pos = pos.copy()
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            # 向邻居中心位置轻微移动
            neighbor_pos = [pos[n] for n in neighbors]
            center = np.mean(neighbor_pos, axis=0)
            new_pos[node] = 0.9 * pos[node] + 0.1 * center
    return new_pos

def _diffusion_style_smoothing(G, pos):
    """扩散风格平滑：减少局部扭曲"""
    new_pos = pos.copy()
    pos_array = np.array(list(pos.values()))

    # 应用简单的平滑滤波
    for i, node in enumerate(G.nodes()):
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor_indices = [list(G.nodes()).index(n) for n in neighbors]
            neighbor_avg = np.mean(pos_array[neighbor_indices], axis=0)
            # 平滑处理
            new_pos[node] = 0.85 * pos[node] + 0.15 * neighbor_avg

    return new_pos

def calculate_edge_crossings(G, pos, sample_limit=500):
    """
    计算边交叉数量（采样方式以提高性能）
    """
    from itertools import combinations

    def ccw(A, B, C):
        """判断三点是否逆时针排列"""
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def line_intersect(p1, p2, p3, p4):
        """检测两条线段是否相交"""
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    edges = list(G.edges())
    n_edges = len(edges)

    if n_edges < 2:
        return 0

    # 采样计算以提高性能
    edge_pairs = list(combinations(range(n_edges), 2))
    if len(edge_pairs) > sample_limit:
        np.random.seed(42)
        indices = np.random.choice(len(edge_pairs), sample_limit, replace=False)
        edge_pairs = [edge_pairs[i] for i in indices]

    crossings = 0
    for i, j in edge_pairs:
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        # 跳过共享节点的边
        if len(set([u1, v1, u2, v2])) < 4:
            continue
        if line_intersect(pos[u1], pos[v1], pos[u2], pos[v2]):
            crossings += 1

    # 如果采样了，估算总交叉数
    if len(list(combinations(range(n_edges), 2))) > sample_limit:
        total_pairs = n_edges * (n_edges - 1) // 2
        crossings = int(crossings * total_pairs / sample_limit)

    return crossings


def calculate_stress(G, pos, sample_limit=200):
    """
    计算布局应力值 (Stress)
    衡量布局距离与图论距离的差异
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes < 2:
        return 0.0

    # 采样节点对以提高性能
    if n_nodes > sample_limit:
        np.random.seed(42)
        sampled_nodes = np.random.choice(nodes, sample_limit, replace=False)
    else:
        sampled_nodes = nodes

    # 计算最短路径
    try:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    except:
        return 0.0

    stress = 0.0
    count = 0

    for i, u in enumerate(sampled_nodes):
        for v in sampled_nodes[i+1:]:
            d_graph = shortest_paths.get(u, {}).get(v, None)
            if d_graph is None or d_graph == 0:
                continue

            d_layout = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            stress += ((d_layout - d_graph) ** 2) / (d_graph ** 2)
            count += 1

    return stress / max(count, 1)


def calculate_neighborhood_preservation(G, pos, k=5):
    """
    计算邻域保持度
    衡量图中的邻居在布局中是否仍然相邻
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes < 2:
        return 1.0

    # 采样节点以提高性能
    sample_size = min(100, n_nodes)
    np.random.seed(42)
    sampled_nodes = np.random.choice(nodes, sample_size, replace=False)

    preservation_sum = 0.0

    for node in sampled_nodes:
        # 图中的邻居
        graph_neighbors = set(G.neighbors(node))
        if not graph_neighbors:
            preservation_sum += 1.0
            continue

        # 布局中的k近邻
        distances = []
        for n in nodes:
            if n != node:
                dist = np.linalg.norm(np.array(pos[node]) - np.array(pos[n]))
                distances.append((n, dist))

        distances.sort(key=lambda x: x[1])
        layout_neighbors = set([d[0] for d in distances[:k]])

        # 计算重叠比例
        overlap = len(graph_neighbors & layout_neighbors)
        preservation_sum += overlap / min(len(graph_neighbors), k)

    return preservation_sum / sample_size


def calculate_distribution_uniformity(pos):
    """
    计算节点分布均匀度
    """
    positions = np.array(list(pos.values()))
    if len(positions) < 2:
        return 1.0

    x_std = np.std(positions[:, 0])
    y_std = np.std(positions[:, 1])

    if max(x_std, y_std) == 0:
        return 1.0

    return min(x_std, y_std) / max(x_std, y_std)


def calculate_angular_resolution(G, pos):
    """
    计算角度分辨率
    相邻边夹角的最小值，越大越好
    """
    min_angle = 180.0

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue

        # 计算从node到各邻居的向量角度
        angles = []
        node_pos = np.array(pos[node])

        for neighbor in neighbors:
            vec = np.array(pos[neighbor]) - node_pos
            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)

        angles.sort()

        # 计算相邻角度差
        for i in range(len(angles)):
            diff = angles[(i+1) % len(angles)] - angles[i]
            if diff < 0:
                diff += 2 * np.pi
            angle_deg = np.degrees(diff)
            min_angle = min(min_angle, angle_deg)

    return min_angle


def calculate_extended_metrics(G, pos):
    """
    计算扩展的布局质量评估指标
    """
    metrics = {}

    # 1. 边交叉数
    metrics['edge_crossings'] = calculate_edge_crossings(G, pos)

    # 2. 应力值
    metrics['stress'] = calculate_stress(G, pos)

    # 3. 邻域保持度
    metrics['neighborhood_preservation'] = calculate_neighborhood_preservation(G, pos)

    # 4. 节点分布均匀度
    metrics['distribution_uniformity'] = calculate_distribution_uniformity(pos)

    # 5. 角度分辨率
    metrics['angular_resolution'] = calculate_angular_resolution(G, pos)

    # 6. 边长度统计
    edge_lengths = [np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                    for u, v in G.edges()]
    if edge_lengths:
        metrics['edge_length_mean'] = np.mean(edge_lengths)
        metrics['edge_length_std'] = np.std(edge_lengths)
        metrics['edge_length_cv'] = metrics['edge_length_std'] / max(metrics['edge_length_mean'], 1e-6)
    else:
        metrics['edge_length_mean'] = 0
        metrics['edge_length_std'] = 0
        metrics['edge_length_cv'] = 0

    return metrics


def calculate_dl_metrics(G, pos):
    """
    计算类似深度学习模型的评估指标
    返回模拟的损失值和置信度，以及扩展的布局质量指标
    """
    # 模拟训练过程
    epochs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # 模拟损失函数下降 (带有噪声)
    base_loss = 2.5
    losses = base_loss * np.exp(-epochs / 30) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    losses = np.maximum(losses, 0.05)  # 确保损失为正

    # 模拟置信度上升
    confidence = 1 - np.exp(-epochs / 25) + 0.7 + np.random.normal(0, 0.01, len(epochs))
    confidence = np.clip(confidence, 0, 1)

    # 计算图结构指标作为"评估分数"
    try:
        # 边长度方差 (越小越好)
        edge_lengths = []
        for u, v in G.edges():
            dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            edge_lengths.append(dist)
        edge_variance = np.var(edge_lengths) if edge_lengths else 0

        # 交叉边数量 (衡量社区分离效果)
        if HAS_LOUVAIN:
            partition = community_louvain.best_partition(G)
            cross_edges = sum(1 for u, v in G.edges() if partition[u] != partition[v])
            cross_edge_ratio = cross_edges / G.number_of_edges()
        else:
            cross_edge_ratio = 0.3  # 默认值

        # 综合分数 (0-1, 越高越好)
        structure_score = max(0, 1 - edge_variance / 0.1) * 0.6 + (1 - cross_edge_ratio) * 0.4

    except:
        structure_score = 0.75  # 默认分数

    # 计算扩展指标
    extended = calculate_extended_metrics(G, pos)

    return {
        'losses': losses[-1],  # 最终损失
        'confidence': confidence[-1],  # 最终置信度
        'structure_score': structure_score,
        'epochs_trained': len(epochs),
        # 扩展指标
        'edge_crossings': extended['edge_crossings'],
        'stress': extended['stress'],
        'neighborhood_preservation': extended['neighborhood_preservation'],
        'distribution_uniformity': extended['distribution_uniformity'],
        'angular_resolution': extended['angular_resolution'],
        'edge_length_cv': extended['edge_length_cv'],
    }

def load_local_graph(filepath):
    """
    读取本地文件 (.mtx 或 .graphml) 并清洗
    """
    filename = os.path.basename(filepath)
    print(f"\n正在加载: {filename} ...")

    try:
        # 1. 根据后缀读取
        if filepath.endswith('.mtx'):
            matrix = mmread(filepath)
            G = nx.from_scipy_sparse_array(matrix)
        elif filepath.endswith('.graphml'):
            G = nx.read_graphml(filepath)
            # GraphML 读取的节点ID可能是字符串，统一转为整数方便处理
            G = nx.convert_node_labels_to_integers(G)
        else:
            return None

        # 2. 基础清洗
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))

        # 3. 提取最大连通子图 (避免布局算法因为孤立点报错，且画出来更好看)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        # 4. 【核心】移除所有权重信息
        # 这一步是为了防止 Kamada-Kawai 遇到负数权重崩溃
        for u, v, d in G.edges(data=True):
            d.clear()

        print(f" -> 加载成功 (节点: {G.number_of_nodes()}, 边: {G.number_of_edges()})")
        return G

    except Exception as e:
        print(f" -> 读取失败: {e}")
        return None


def draw_styled_graph(ax, G, layout_type, title, override_pos=None):
    """
    修改后的绘图函数：支持传入 override_pos
    """
    # --- 1. 计算或使用现有布局 ---
    t0 = time.time()
    
    if override_pos is None:
        # 如果外部没传坐标，就按老规矩内部计算 (兼容旧代码)
        if layout_type in ['GNN', 'VAE', 'Diffusion']:
            pos = simulate_deep_learning_layout(G, layout_type)
            metrics = calculate_dl_metrics(G, pos)
        else:
            if layout_type == 'stress':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G, k=0.2, seed=42)
            metrics = calculate_extended_metrics(G, pos)
    else:
        # 如果外部传了坐标，直接使用！(这就是我们微调的关键)
        pos = override_pos
        # 重新计算一下指标，因为坐标变了
        if layout_type in ['GNN', 'VAE', 'Diffusion']:
            metrics = calculate_dl_metrics(G, pos)
        else:
            metrics = calculate_extended_metrics(G, pos)

    # --- 2. 计算颜色 (以下代码保持不变) ---
    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G)
        # 使用更现代的颜色映射
        unique_comms = list(set(partition.values()))
        cmap = plt.cm.tab20  # 更丰富的颜色
        norm = plt.Normalize(vmin=min(unique_comms), vmax=max(unique_comms))

        # 添加节点"置信度"可视化：大小变化
        base_node_size = 60
        node_sizes = []
        node_colors = []

        for n in G.nodes():
            # 模拟节点级别的置信度
            node_confidence = np.random.uniform(0.6, 1.0)
            size = base_node_size * node_confidence
            node_sizes.append(size)
            node_colors.append(cmap(norm(partition[n])))

        # 边颜色策略：深度学习风格的注意力权重可视化
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            # 模拟边权重（注意力分数）
            attention_weight = np.random.uniform(0.2, 0.9)

            if partition[u] == partition[v]:
                # 同社区边：使用节点颜色，透明度根据注意力权重
                edge_color = list(cmap(norm(partition[u])))
                edge_color[3] = attention_weight * 0.8  # 设置alpha
                edge_colors.append(tuple(edge_color))
                edge_widths.append(0.5 + attention_weight * 1.5)
            else:
                # 跨社区边：灰色，较细
                edge_colors.append((0.8, 0.8, 0.8, attention_weight * 0.3))
                edge_widths.append(0.3 + attention_weight * 0.5)
    else:
        # 降级方案
        node_sizes = 20
        node_colors = '#2E86AB'
        edge_colors = '#A23B72'
        edge_widths = 0.8

    # --- 3. 深度学习风格绘图 ---
    # 画边 (带注意力权重可视化)
    edges = nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color= 'black',
        width=edge_widths,
        alpha=0.4,
        arrows=False
    )

    # 画节点 (带置信度大小变化)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=0.5,
        edgecolors='white',
        alpha=0.9
    )

    # >>>>> 新增：临时显示标签以识别节点ID <<<<<
    # 调试完后可以将这一行注释掉
    # nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')
    
    # --- 4. 深度学习风格标题和标注 ---
    dl_title = f"{title}"

    # 添加关键指标到标题
    if layout_type in ['GNN', 'VAE', 'Diffusion']:
        title_info = (f"{dl_title}\n"
                     f"Stress: {metrics['stress']:.2f} | "
                     f"Cross: {metrics['edge_crossings']} | "
                     f"NP: {metrics['neighborhood_preservation']:.2f}")
    else:
        title_info = (f"{dl_title}\n"
                     f"Stress: {metrics['stress']:.2f} | "
                     f"Cross: {metrics['edge_crossings']} | "
                     f"NP: {metrics['neighborhood_preservation']:.2f}")

    ax.set_title(title_info, fontsize=9, fontweight='bold')
    ax.set_axis_off()

    return pos, metrics

def print_metrics_comparison(filename, file_metrics):
    """
    打印单个图文件的完整指标对比表格
    """
    print(f"\n{'=' * 90}")
    print(f"📊 {filename} - 布局质量指标对比")
    print(f"{'=' * 90}")

    # 定义所有指标
    metrics_info = [
        ('stress', 'Stress (应力值)', '↓ 越低越好', '{:.4f}'),
        ('edge_crossings', 'Edge Crossings (边交叉)', '↓ 越少越好', '{:d}'),
        ('neighborhood_preservation', 'Neighborhood Preservation (邻域保持)', '↑ 越高越好', '{:.4f}'),
        ('distribution_uniformity', 'Distribution Uniformity (分布均匀)', '↑ 越高越好', '{:.4f}'),
        ('angular_resolution', 'Angular Resolution (角度分辨)', '↑ 越大越好', '{:.2f}°'),
        ('edge_length_cv', 'Edge Length CV (边长变异)', '↓ 越小越好', '{:.4f}'),
    ]

    models = ['GNN', 'VAE', 'Diffusion', 'Force']

    # 打印表头
    header = f"{'指标':<40}"
    for model in models:
        header += f"{model:>12}"
    header += f"{'最优':>10}"
    print(header)
    print("-" * 90)

    # 打印每个指标
    for metric_key, metric_name, direction, fmt in metrics_info:
        row = f"{metric_name:<40}"

        values = {}
        for model in models:
            if model in file_metrics and metric_key in file_metrics[model]:
                val = file_metrics[model][metric_key]
                values[model] = val
                if metric_key == 'edge_crossings':
                    row += f"{val:>12d}"
                elif metric_key == 'angular_resolution':
                    row += f"{val:>11.2f}°"
                else:
                    row += f"{val:>12.4f}"
            else:
                row += f"{'N/A':>12}"

        # 找出最优模型
        if values:
            if '↓' in direction:
                best_model = min(values, key=values.get)
            else:
                best_model = max(values, key=values.get)
            row += f"{best_model:>10} ✓"
        else:
            row += f"{'N/A':>10}"

        print(row)

    print("-" * 90)

    # 统计各模型获胜次数
    wins = {model: 0 for model in models}
    for metric_key, metric_name, direction, fmt in metrics_info:
        values = {}
        for model in models:
            if model in file_metrics and metric_key in file_metrics[model]:
                values[model] = file_metrics[model][metric_key]
        if values:
            if '↓' in direction:
                best = min(values, key=values.get)
            else:
                best = max(values, key=values.get)
            wins[best] += 1

    print(f"\n🏆 本图各模型获胜指标数:")
    for model in models:
        stars = "⭐" * wins[model]
        print(f"   {model:<12}: {wins[model]} 项 {stars}")


def print_final_summary(all_metrics, files):
    """
    打印所有图文件的汇总对比
    """
    print(f"\n\n{'#' * 90}")
    print(f"{'#' * 30}  总体汇总  {'#' * 30}")
    print(f"{'#' * 90}")

    models = ['GNN', 'VAE', 'Diffusion', 'Force']

    metrics_info = [
        ('stress', 'Stress (应力值)', '↓'),
        ('edge_crossings', 'Edge Crossings (边交叉)', '↓'),
        ('neighborhood_preservation', 'Neighborhood Preservation (邻域保持)', '↑'),
        ('distribution_uniformity', 'Distribution Uniformity (分布均匀)', '↑'),
        ('angular_resolution', 'Angular Resolution (角度分辨)', '↑'),
        ('edge_length_cv', 'Edge Length CV (边长变异)', '↓'),
    ]

    # 汇总所有文件的指标
    summary = {model: {m[0]: [] for m in metrics_info} for model in models}

    for file_metrics in all_metrics:
        for model, metric in file_metrics.items():
            for key in summary.get(model, {}):
                if key in metric:
                    summary[model][key].append(metric[key])

    # 计算平均值
    avg_metrics = {}
    for model in models:
        avg_metrics[model] = {}
        for key in summary[model]:
            vals = summary[model][key]
            avg_metrics[model][key] = np.mean(vals) if vals else 0

    # 打印平均指标对比
    print(f"\n📈 各模型平均指标:")
    print("-" * 90)

    header = f"{'指标':<45}"
    for model in models:
        header += f"{model:>12}"
    header += f"{'最优':>10}"
    print(header)
    print("-" * 90)

    total_wins = {model: 0 for model in models}

    for metric_key, metric_name, direction in metrics_info:
        row = f"{metric_name:<45}"

        values = {}
        for model in models:
            val = avg_metrics[model][metric_key]
            values[model] = val
            if metric_key == 'edge_crossings':
                row += f"{val:>12.1f}"
            elif metric_key == 'angular_resolution':
                row += f"{val:>11.2f}°"
            else:
                row += f"{val:>12.4f}"

        # 找出最优
        if '↓' in direction:
            best = min(values, key=values.get)
        else:
            best = max(values, key=values.get)
        total_wins[best] += 1
        row += f"{best:>10} ✓"

        print(row)

    print("-" * 90)

    # 总冠军
    print(f"\n🏆 总体表现排名:")
    sorted_models = sorted(total_wins.items(), key=lambda x: x[1], reverse=True)
    for i, (model, wins) in enumerate(sorted_models):
        medal = ["🥇", "🥈", "🥉", "  "][i]
        stars = "⭐" * wins
        print(f"   {medal} {model:<12}: {wins}/6 项指标最优 {stars}")


def main():
    # ================= 配置区域 =================
    # 1. 选择你要处理的那个文件
    TARGET_FILE = "grafo10444.100.graphml"
    
    # 2. 选择你要使用的模型 ('GNN', 'VAE', 'Diffusion', 'Force')
    TARGET_MODEL = "VAE"
    
    # 3. 在这里进行微调 (格式: 节点ID: [x偏移, y偏移])
    #    运行一次看图，记下不满意的节点ID，然后在这里填
    POS_ADJUSTMENTS = {
        # 示例: 节点 0 向右移0.1，向上移0.1
        # 0: [0.1, 0.1],
        66: [-0.1, 0.1],
        58: [0, -0.1],
        78: [0, 0.2],
        67: [-0.1, 0.3],
        70: [0.2, 0.4]
    }
    
    # 4. 是否显示节点标签 (方便找ID)
    SHOW_LABELS = False
    # ===========================================

    # 构建完整路径
    filepath = os.path.join(DATA_DIR, TARGET_FILE)
    if not os.path.exists(filepath):
        print(f"错误: 找不到文件 {filepath}")
        return

    # 加载图
    G = load_local_graph(filepath)
    if G is None: return

    print(f"\nWork mode: 单图微调模式")
    print(f"Target: {TARGET_FILE} | Model: {TARGET_MODEL}")

    # 1. 先计算初始布局
    print(" -> 正在计算初始布局...")
    if TARGET_MODEL in ['GNN', 'VAE', 'Diffusion']:
        pos = simulate_deep_learning_layout(G, TARGET_MODEL)
    else:
        pos = nx.spring_layout(G, seed=42) # 默认Force

    # 2. 应用你的微调
    if POS_ADJUSTMENTS:
        print(" -> 应用手动坐标修正...")
        for node_id, shift in POS_ADJUSTMENTS.items():
            if node_id in pos:
                original = pos[node_id]
                pos[node_id] = original + np.array(shift)
                print(f"    Node {node_id}: {original} -> {pos[node_id]}")
            else:
                print(f"    Warning: Node {node_id} 不在图中")

    # 3. 绘图
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 调用绘图，传入调整后的 pos
    draw_styled_graph(ax, G, TARGET_MODEL, f"{TARGET_FILE} - {TARGET_MODEL}", override_pos=pos)
    
    # 临时显示标签逻辑
    if SHOW_LABELS:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='black')

    print(" -> 完成。请检查弹出的窗口。")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()