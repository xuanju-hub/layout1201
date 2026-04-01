import os
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import networkx as nx
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
                if d < min_dist: overlap_pairs.append((d, u, v))
                else: non_overlap_pairs.append((d, u, v))
        overlap_pairs.sort(key=lambda x: x[0])
        non_overlap_pairs.sort(key=lambda x: x[0])
        return overlap_pairs, non_overlap_pairs

    overlap_pairs, non_overlap_pairs = collect_pairs()
    current = len(overlap_pairs)
    if current == target_overlaps: return pos

    diff = abs(target_overlaps - current)
    base = max(1, target_overlaps if target_overlaps > 0 else current)
    diff_ratio = min(1.0, diff / base)

    if current < target_overlaps:
        need = target_overlaps - current
        changed = 0
        for d, u, v in non_overlap_pairs:
            if changed >= need: break
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
            if changed >= need: break
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

dir8_map = {
    'can_838.mtx': [0.0]*8,
    'grafo9873.35.graphml': [0.0]*8,
}
target_overlaps_map = {
    'grafo9873.35.graphml': 4,
    'can_838.mtx': 30
}

# 1. 载入原始图数据并生成初始布局
file_name = 'grafo4323.78.graphml'
file_path = os.path.join(DATASET_DIR, file_name)

if file_name.endswith('.mtx'): G = read_mtx(file_path)
elif file_name.endswith('.graphml'): G = read_graphml(file_path)

G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

seed = stable_seed_from_name(file_name)
init_pos = apply_fa2(G, seed=seed)

node_rad = 0.5
if file_name in target_overlaps_map:
    init_pos = enforce_overlaps(G, init_pos, target_overlaps_map[file_name], node_rad)

# 2. 生成初始Cytoscape元素 (颜色稍后在Dash回调中由聚类逻辑赋予)
elements = []
for node in G.nodes():
    x, y = init_pos[node]
    elements.append({
        'data': {'id': str(node), 'label': str(node), 'color': '#999999'},
        'position': {'x': float(x * 100), 'y': float(-y * 100)},
        'classes': 'node-cls'
    })
for edge in G.edges():
    elements.append({
        'data': {'source': str(edge[0]), 'target': str(edge[1])}
    })

# 3. 初始化 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2(f"交互式图聚类与布局 - {file_name}"),
    
    html.Div([
        html.Label("布局模式:"),
        dcc.RadioItems(
            id='cluster-mode',
            options=[
                {'label': '原始布局', 'value': False},
                {'label': '启用聚类', 'value': True}
            ],
            value=True,
            inline=True,
            style={'marginBottom': '10px'}
        ),
        html.Label("Louvain 聚类分辨率 (Resolution):"),
        dcc.Slider(
            id='resolution-slider',
            min=0.1, max=3.0, step=0.1, value=1.0,
            marks={i/10: str(i/10) for i in range(1, 31, 5)}
        )
    ], style={'padding': '20px', 'width': '50%'}),
    
    html.Button("保存当前布局为图片", id="save-btn", n_clicks=0, style={'marginBottom': '10px', 'padding': '10px'}),
    html.Div(id="output-msg", style={'color': 'green', 'marginBottom': '10px'}),
    
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        layout={'name': 'preset'}, 
        style={'width': '100%', 'height': '800px', 'border': '1px solid #ccc'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'width': '20px', 'height': '20px',
                    'background-color': 'data(color)', # 动态获取颜色
                    'border-width': '1px', 'border-color': 'black'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 1, 'line-color': '#999', 'opacity': 0.5
                }
            }
        ]
    )
])

# 生成颜色调色板
cmap = plt.cm.get_cmap('tab20', 20)
def get_color(idx):
    return mcolors.to_hex(cmap(idx % 20))

# 4. 回调函数：动态聚类 & 保存图
@app.callback(
    [Output('cytoscape-graph', 'elements'),
     Output("output-msg", "children")],
    [Input('cluster-mode', 'value'),
     Input('resolution-slider', 'value'),
     Input("save-btn", "n_clicks")],
    [State('cytoscape-graph', 'elements')]
)
def update_and_save(cluster_mode, resolution, n_clicks, current_elements):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'resolution-slider'
    
    # 如果只是点击了保存按钮，我们只负责保存图片，不重新计算和刷新网页的 elements
    if triggered_id == 'save-btn' and n_clicks > 0:
        new_pos = {}
        color_list_for_mpl = []
        node_order = []
        
        for ele in current_elements:
            if 'position' in ele:
                node_id = ele['data']['id']
                n_obj = int(node_id) if node_id.isdigit() else node_id
                
                new_pos[n_obj] = (ele['position']['x'], -ele['position']['y'])
                node_order.append(n_obj)
                color_list_for_mpl.append(ele['data']['color'])
        
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        
        nx.draw_networkx_nodes(G, new_pos, nodelist=node_order, node_size=30, node_color=color_list_for_mpl, alpha=0.9, edgecolors='black', linewidths=0.6)
        nx.draw_networkx_edges(G, new_pos, alpha=0.15, width=0.5)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename_suffix = f"clustered_res{resolution}" if cluster_mode else f"clustered_res{resolution}_original"
        out_path = os.path.join(OUTPUT_DIR, f"{file_name}_{filename_suffix}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # dash.no_update 告诉前端不要刷新 Cytoscape 的 elements
        return dash.no_update, f"图片已成功保存至: {out_path}"

    # --- 以下是滑动条或单选框触发的更新逻辑 ---
    
    # 添加 seed=seed，保证给定的图和 resolution 总是产生确定的聚类结果，不会随机变动
    communities = nx.community.louvain_communities(G, resolution=resolution, seed=seed)
    
    node_comm = {}
    for i, comm in enumerate(communities):
        color = get_color(i)
        for node in comm:
            node_comm[node] = {'id': i, 'color': color}

    if cluster_mode:
        global_cx = sum(pos[0] for pos in init_pos.values()) / len(init_pos)
        global_cy = sum(pos[1] for pos in init_pos.values()) / len(init_pos)

        explode_factor = 2.0  
        pull_strength = 0.7   

        comm_centers = {}
        for i, comm in enumerate(communities):
            cx = sum(init_pos[n][0] for n in comm) / len(comm)
            cy = sum(init_pos[n][1] for n in comm) / len(comm)
            
            new_cx = global_cx + (cx - global_cx) * explode_factor
            new_cy = global_cy + (cy - global_cy) * explode_factor
            comm_centers[i] = (new_cx, new_cy)

    updated_elements = []
    for ele in current_elements:
        new_ele = ele.copy()
        if 'position' in new_ele:
            node_id = new_ele['data']['id']
            n_obj = int(node_id) if node_id.isdigit() else node_id
            
            orig_x, orig_y = init_pos[n_obj]
            
            if n_obj in node_comm:
                new_ele['data']['color'] = node_comm[n_obj]['color']
                c_id = node_comm[n_obj]['id']
            else:
                new_ele['data']['color'] = '#999999'

            if cluster_mode and n_obj in node_comm:
                cx, cy = comm_centers[c_id]
                new_x = orig_x + (cx - orig_x) * pull_strength
                new_y = orig_y + (cy - orig_y) * pull_strength
                
                new_ele['position'] = {
                    'x': float(new_x * 100),
                    'y': float(-new_y * 100)
                }
            else:
                new_ele['position'] = {
                    'x': float(orig_x * 100),
                    'y': float(-orig_y * 100)
                }
        updated_elements.append(new_ele)
        
    status_text = f"当前聚类数: {len(communities)} (Resolution: {resolution})"
    status_text += " - 聚拢布局" if cluster_mode else " - 原始布局"
        
    return updated_elements, status_text

if __name__ == '__main__':
    print("启动 Dash 聚类交互应用: http://127.0.0.1:8050")
    app.run(debug=True)