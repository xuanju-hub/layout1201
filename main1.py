import networkx as nx
from pyvis.network import Network
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import community.community_louvain as community_louvain
from scipy.io import mmread
import numpy as np

# --- 1. 配置区域 (你可以在这里随心所欲地改) ---
BACKGROUND_COLOR = "#ffffff"  # 背景颜色：白色 (#ffffff) 或 深色 (#222222)
FONT_COLOR = "black"  # 字体颜色：跟背景反着来
NODE_SIZE = 12  # 节点大小
EDGE_COLOR = "#cccccc"  # 边颜色 (浅灰)

# 尝试导入社区检测
try:
    import community.community_louvain as community_louvain

    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def get_hex_colors_from_cmap(n, cmap_name='Spectral'):
    """生成 n 个不一样的 Hex 颜色代码"""
    cmap = plt.cm.get_cmap(cmap_name, n)
    # 将 Matplotlib 的 RGBA 转为 Hex (例如 #FF0000)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


def get_hex_colors(n, cmap_name='jet'):
    """生成 n 个不一样的 Hex 颜色代码"""
    cmap = matplotlib.colormaps[cmap_name].resampled(n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


def apply_clustering(G, method='louvain'):
    """
    计算聚类并给节点分配颜色
    method: 'louvain' (推荐，快) 或 'girvan_newman' (论文原版，慢)
    """
    print(f"正在计算聚类 (算法: {method})...")

    partition = {}  # 格式: {节点ID: 社区ID}

    if method == 'louvain':
        # === 方案 A: Louvain (秒级完成，推荐) ===
        try:
            partition = community_louvain.best_partition(G)
        except Exception as e:
            print(f"Louvain 计算失败: {e}")
            return

    elif method == 'girvan_newman':
        # === 方案 B: Girvan-Newman (论文原版 ，但在大图上极慢) ===
        # 这个算法通过不断移除边来分裂社区
        try:
            # 只取第一层分裂结果（通常分裂成 2-5 个大社区）
            comp = nx.community.girvan_newman(G)
            # next(comp) 返回的是第一层分裂的节点集合元组
            first_level_communities = next(comp)

            # 将集合转换为字典格式 {node: community_id}
            for comm_id, nodes in enumerate(first_level_communities):
                for node in nodes:
                    partition[node] = comm_id
        except Exception as e:
            print(f"Girvan-Newman 计算失败 (可能图太大): {e}")
            return

    # --- 上色步骤 ---
    # 1. 统计有多少个社区
    if not partition:
        # 如果聚类失败，全染成蓝色
        for node in G.nodes():
            G.nodes[node]['group'] = 0
            G.nodes[node]['color'] = "#4e79a7"
        return

    num_comms = len(set(partition.values()))
    print(f"检测到 {num_comms} 个社区")

    # 2. 生成色盘
    palette = get_hex_colors(num_comms, 'jet')

    # 3. 赋值给 PyVis 需要的属性
    for node, comm_id in partition.items():
        G.nodes[node]['group'] = comm_id  # 用于 PyVis 逻辑分组
        G.nodes[node]['color'] = palette[comm_id]  # 强制指定颜色
        G.nodes[node]['title'] = f"Node: {node}<br>Group: {comm_id}"  # 鼠标悬停显示
def load_and_prep_graph(filepath):
    """读取并预处理图 (添加颜色属性)"""
    filename = os.path.basename(filepath)
    print(f"正在加载并处理: {filename} ...")

    # A. 读取
    if filepath.endswith('.mtx'):
        G = nx.from_scipy_sparse_array(mmread(filepath))
    elif filepath.endswith('.graphml'):
        G = nx.read_graphml(filepath)
        G = nx.convert_node_labels_to_integers(G)
    else:
        return None

    # B. 清洗
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    # 取最大连通子图 (为了展示效果好)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # C. 【关键】计算并设置节点颜色
    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G)
        num_comms = len(set(partition.values()))
        # 生成对应数量的颜色列表
        palette = get_hex_colors_from_cmap(num_comms, 'jet')  # 'jet', 'viridis', 'coolwarm' 都可以

        for node in G.nodes():
            comm_id = partition[node]
            # --- 核心：直接把颜色写进节点属性 ---
            G.nodes[node]['color'] = palette[comm_id]
            G.nodes[node]['size'] = NODE_SIZE
            G.nodes[node]['title'] = f"Node: {node}\nGroup: {comm_id}"  # 鼠标悬停提示
    else:
        # 如果没装社区库，就统用蓝色
        for node in G.nodes():
            G.nodes[node]['color'] = "#4e79a7"
            G.nodes[node]['size'] = NODE_SIZE
    if G.number_of_nodes() > 200:
        apply_clustering(G, method='louvain')
    else:
        # 只有小图才尝试论文原版算法，防止卡死
        apply_clustering(G, method='girvan_newman')

        # 设置节点大小 (可选)
    for node in G.nodes():
        G.nodes[node]['size'] = NODE_SIZE
    return G


def generate_interactive_html(filepath):
    G = load_and_prep_graph(filepath)
    if G is None or G.number_of_nodes() == 0:
        print("图为空或读取失败")
        return

    name = os.path.basename(filepath).split('.')[0]

    # === 初始化 PyVis (关键修改：使用 local 模式) ===
    # local 模式会在当前目录下生成一个 'lib' 文件夹，
    # 里面包含 vis.js 等文件，既不用联网，也不会报错。
    net = Network(
        height="800px",
        width="100%",
        bgcolor=BACKGROUND_COLOR,
        font_color=FONT_COLOR,
        notebook=False,
        select_menu=True,
        cdn_resources='local'  # <--- 核心修改：改为 local
    )

    net.from_nx(G)

    # 设置边颜色
    for edge in net.edges:
        edge['color'] = EDGE_COLOR
        edge['width'] = 0.5

    # === 设置物理参数 (直线 + 强排斥) ===
    net.set_options("""
    {
      "edges": {
        "smooth": false
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -2000,
          "springLength": 200,
          "damping": 0.8
        },
        "solver": "forceAtlas2Based"
      }
    }
    """)

    output_file = f"{name}_interactive.html"
    print(f"  - 正在生成 HTML: {output_file}")
    try:
        net.show(output_file, notebook=False)
        print(f"✅ 成功！请打开 {output_file} 查看。")
        print("   (注意：请确保生成的 'lib' 文件夹与 HTML 在同一目录下)")
    except Exception as e:
        print(f"❌ 生成失败: {e}")

if __name__ == "__main__":
    # 指定你要画的文件路径
    target_file = "datasets/grafo9873.35.graphml"

    if os.path.exists(target_file):
        generate_interactive_html(target_file)
    else:
        print(f"文件不存在: {target_file}，请检查 datasets 目录")
