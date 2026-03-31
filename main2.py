import networkx as nx
from pyvis.network import Network
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import os
import math
from scipy.io import mmread
import warnings

warnings.filterwarnings("ignore")

# --- 配置 ---
BACKGROUND_COLOR = "#ffffff"
FONT_COLOR = "black"


def get_hex_colors(n, cmap_name='jet'):
    """生成颜色"""
    try:
        cmap = matplotlib.colormaps[cmap_name].resampled(n)
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name, n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


def grouped_layout(G, partition, k_repulsion=0.5):
    """
    【核心算法】分块布局：将同一社区的节点强制聚在一起
    """
    pos = {}
    communities = {}

    # 1. 将节点按社区分组
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    num_communities = len(communities)

    # 2. 计算每个社区“中心点”的坐标 (把它们排成一个大圆圈)
    # 社区越多，圆圈越大
    radius = math.sqrt(num_communities) * 2.0
    center_positions = {}
    sorted_comm_ids = sorted(communities.keys())

    for i, comm_id in enumerate(sorted_comm_ids):
        angle = 2 * math.pi * i / num_communities
        center_x = radius * math.cos(angle)
        center_y = radius * math.sin(angle)
        center_positions[comm_id] = (center_x, center_y)

    # 3. 分别计算每个社区内部的布局，并移动到对应的中心点
    print(f"正在分别计算 {num_communities} 个社区的局部布局...")

    for comm_id, nodes in communities.items():
        # 提取子图
        subg = G.subgraph(nodes)

        # 计算子图布局 (力导向)
        # k 值越小，社区内部缩得越紧
        sub_pos = nx.spring_layout(subg, k=k_repulsion, seed=42)

        # 将子图坐标平移到该社区的中心位置
        cx, cy = center_positions[comm_id]
        for node, (x, y) in sub_pos.items():
            # final_x = 社区中心X + 局部x
            pos[node] = (cx + x, cy + y)

    return pos


def generate_cluster_html(filepath):
    filename = os.path.basename(filepath)
    print(f"正在处理: {filename} ...")

    # 1. 读取数据
    if filepath.endswith('.mtx'):
        G = nx.from_scipy_sparse_array(mmread(filepath))
    elif filepath.endswith('.graphml'):
        G = nx.read_graphml(filepath)
        G = nx.convert_node_labels_to_integers(G)
    else:
        return

    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # 2. 计算社区
    print("正在检测社区...")
    try:
        partition = community_louvain.best_partition(G)
    except:
        partition = {n: 0 for n in G.nodes()}  # 失败则全为0

    # 3. 【关键】计算聚类布局坐标
    # 这一步计算出了每个点应该在的位置 (x, y)
    fixed_positions = grouped_layout(G, partition)

    # 4. 设置 PyVis 属性
    num_comms = len(set(partition.values()))
    palette = get_hex_colors(num_comms, 'jet')

    for node in G.nodes():
        comm_id = partition[node]
        G.nodes[node]['group'] = comm_id
        G.nodes[node]['color'] = palette[comm_id]
        G.nodes[node]['size'] = 15

        # --- 核心：将计算好的坐标写入节点属性 ---
        # PyVis 读取 x, y 后会把节点钉在那个位置
        x, y = fixed_positions[node]
        G.nodes[node]['x'] = x * 100  # 放大坐标，防止太挤
        G.nodes[node]['y'] = y * 100

    # 5. 生成 HTML
    net = Network(
        height="800px", width="100%",
        bgcolor=BACKGROUND_COLOR, font_color=FONT_COLOR,
        notebook=False,
        cdn_resources='local'  # 记得用 local 模式
    )

    net.from_nx(G)

    # 6. 关闭物理引擎 (因为我们已经算好位置了，不需要物理引擎再乱动)
    # 或者开启物理引擎但设置极强的阻尼
    net.toggle_physics(False)

    output_file = f"{filename.split('.')[0]}_clustered.html"
    net.show(output_file, notebook=False)
    print(f"✅ 聚类布局完成！请打开: {output_file}")


if __name__ == "__main__":
    # 换成你的文件名
    target = "datasets/grafo9873.35.graphml"
    if os.path.exists(target):
        generate_cluster_html(target)
    else:
        print("文件不存在")