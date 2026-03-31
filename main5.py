import os
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import networkx as nx
from scipy.io import mmread
import matplotlib.pyplot as plt
import random
import numpy as np
import hashlib
import math
from fa2 import ForceAtlas2

DATASET_DIR = "/Users/juxuan/PycharmProjects/layout1201/datasets"
OUTPUT_DIR = "/Users/juxuan/PycharmProjects/layout1201/output_images"

def read_mtx(file_path):
    mat = mmread(file_path)
    return nx.from_scipy_sparse_array(mat)

def read_graphml(file_path):
    return nx.read_graphml(file_path)

def apply_fa2(G, seed=42):
    print(f"Applying ForceAtlas2 to graph with {G.number_of_nodes()} nodes...")
    random.seed(seed)
    np.random.seed(seed)
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
    return forceatlas2.forceatlas2_networkx_layout(G, pos=init_pos, iterations=100)

def stable_seed_from_name(name: str, base: int = 2026) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base) % (2**31 - 1)

def enforce_overlaps(G, pos, target_overlaps, node_radius):
    nodes = sorted(G.nodes(), key=str)
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
        overlap_pairs.sort(key=lambda x: x[0])
        non_overlap_pairs.sort(key=lambda x: x[0])
        return overlap_pairs, non_overlap_pairs

    overlap_pairs, non_overlap_pairs = collect_pairs()
    current = len(overlap_pairs)
    print(f"  [微调] 当前重叠数: {current}, 目标重叠数: {target_overlaps}")

    if current == target_overlaps:
        return pos

    diff = abs(target_overlaps - current)
    base = max(1, target_overlaps if target_overlaps > 0 else current)
    diff_ratio = min(1.0, diff / base)

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
            if dist == 0: continue
            target_d = min_dist * (0.95 - 0.45 * diff_ratio)
            target_d = max(0.05 * min_dist, target_d)
            if dist > target_d:
                move = dist - target_d
                nx_, ny_ = dx / dist, dy / dist
                pos[v] = (vx + nx_ * move, vy + ny_ * move)
                changed += 1
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
                dx, dy = 1.0, 0.0
                dist = 1.0
            nx_, ny_ = dx / dist, dy / dist
            target_d = min_dist * (1.01 + 1.8 * diff_ratio)
            pos[v] = (ux + nx_ * target_d, uy + ny_ * target_d)
            changed += 1
    return pos

def apply_8dir_radial_warp(pos, dir_strengths, power=2.0):
    if not pos: return pos
    if len(dir_strengths) != 8:
        raise ValueError("dir_strengths 必须是8个方向参数")
    nodes = list(pos.keys())
    cx = sum(pos[n][0] for n in nodes) / len(nodes)
    cy = sum(pos[n][1] for n in nodes) / len(nodes)
    r_max = 0.0
    for n in nodes:
        x, y = pos[n]
        r = math.hypot(x - cx, y - cy)
        r_max = max(r_max, r)
    if r_max == 0: return pos
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
        if scale < 0.05: scale = 0.05
        new_pos[n] = (cx + dx * scale, cy + dy * scale)
    return new_pos

dir8_map = {
    'plskz362.mtx':            [0.0, 1.00, 0.0, 3.00, 1.00, 0.0, 1.80, 0.80],
    'can_838.mtx':             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'grafo4323.78.graphml':    [0.00, 0.0, 0.00, 1.60, 1.00, 0.00, -0.50, 0.00],
    'grafo9873.35.graphml':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

target_overlaps_map = {
    'grafo4323.78.graphml': 16,
    'grafo9873.35.graphml': 4,
    'plskz362.mtx': 16,
    'can_838.mtx':  30
}

# 1. 载入原始图数据并生成初始布局
# 这里以 can_838.mtx 为例，你可以根据需求更改
file_name = 'grafo9873.35.graphml'
file_path = os.path.join(DATASET_DIR, file_name)

if file_name.endswith('.mtx'):
    G = read_mtx(file_path)
elif file_name.endswith('.graphml'):
    G = read_graphml(file_path)
else:
    raise ValueError("不支持的文件格式")

# 提取最大连通子图以方便展示，或者直接使用全图
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

print("正在计算初始布局...")
seed = stable_seed_from_name(file_name)
init_pos = apply_fa2(G, seed=seed)

node_rad = 0.5
if file_name in target_overlaps_map:
    target = target_overlaps_map[file_name]
    print(f"[{file_name}] 开始调整重叠数至目标: {target}")
    init_pos = enforce_overlaps(G, init_pos, target_overlaps=target, node_radius=node_rad)

dir_strengths = dir8_map.get(file_name)
if dir_strengths is not None:
    init_pos = apply_8dir_radial_warp(init_pos, dir_strengths=dir_strengths, power=2.0)

# 2. 转换为 Cytoscape 元素格式
elements = []
for node in G.nodes():
    x, y = init_pos[node]
    elements.append({
        'data': {'id': str(node), 'label': str(node)},
        # 将传入 Cytoscape 的 y 坐标取负，使其在网页上显示方向与 matplotlib 保持一致
        'position': {'x': float(x * 100), 'y': float(-y * 100)} 
    })

for edge in G.edges():
    elements.append({
        'data': {'source': str(edge[0]), 'target': str(edge[1])}
    })

# 3. 初始化 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2(f"交互式图布局调整 - {file_name}"),
    html.Button("保存当前布局为图片", id="save-btn", n_clicks=0, style={'marginBottom': '10px', 'padding': '10px'}),
    html.Div(id="output-msg", style={'color': 'green', 'marginBottom': '10px'}),
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        layout={'name': 'preset'}, # 使用 preset 以应用我们计算的初始坐标
        style={'width': '100%', 'height': '800px', 'border': '1px solid #ccc'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'width': '30px',  # 放大节点宽度
                    'height': '30px', # 放大节点高度
                    'background-color': 'skyblue',
                    'border-width': '1px',
                    'border-color': 'black'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 1,
                    'line-color': '#666',  # 加深边的颜色，例如从 #ccc 改为 #666
                    'opacity': 0.8         # 提高不透明度，使颜色更明显
                }
            }
        ]
    )
])

# 4. 回调函数：处理保存按钮点击事件
@app.callback(
    Output("output-msg", "children"),
    Input("save-btn", "n_clicks"),
    State('cytoscape-graph', 'elements')
)
def save_layout_to_image(n_clicks, current_elements):
    if n_clicks == 0 or not current_elements:
        return ""
    
    # 从网页元素中提取更新后的坐标
    new_pos = {}
    for ele in current_elements:
        if 'position' in ele: # 这是一个节点
            node_id = ele['data']['id']
            # 根据原始类型转换 node_id（这里假设所有节点 ID 在 networkx 中是 int）
            try:
                n = int(node_id)
            except:
                n = node_id
            
            # 反转 Y 轴以匹配 Matplotlib 坐标系
            new_pos[n] = (ele['position']['x'], -ele['position']['y'])
    
    # 使用 Matplotlib 将新布局保存为图片
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    nx.draw_networkx_nodes(G, new_pos, node_size=15, node_color='skyblue', alpha=0.8, edgecolors='black', linewidths=0.2)
    nx.draw_networkx_edges(G, new_pos, alpha=0.15, width=0.5)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{file_name}_manual_adjusted.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"图片已成功保存至: {out_path}"

if __name__ == '__main__':
    print("启动 Web 服务器, 请在浏览器中打开 http://127.0.0.1:8050")
    app.run(debug=True)