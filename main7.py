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

def apply_top_down_flow(G, pos, flow_strength=1.0):
    """根据 BFS 深度将节点从上到下排布，flow_strength 控制混合程度(0.0~1.0)"""
    if not G.nodes or flow_strength == 0:
        return pos
    
    # 找到度最大的节点作为"源头"（最上层节点）
    root = max(G.degree, key=lambda x: x[1])[0]
    
    # 计算所有节点到源节点的层级深度
    levels = nx.single_source_shortest_path_length(G, root)
    max_depth = max(levels.values()) if levels else 1
    if max_depth == 0: max_depth = 1
    
    # 获取原有布局的 Y 轴跨度，以匹配缩放比例
    ys = [p[1] for p in pos.values()]
    min_y, max_y = min(ys), max(ys)
    y_span = max_y - min_y if max_y > min_y else 100
    
    new_pos = {}
    for node in pos:
        x, old_y = pos[node]
        # 获取深度，如果不连通则放在最底层
        depth = levels.get(node, max_depth)
        
        # 计算目标 Y 坐标（最上层是 max_y，往下依次减小）
        target_y = max_y - (depth / max_depth) * y_span
        
        # 混合原始 Y 坐标与目标层次 Y 坐标
        new_y = old_y * (1 - flow_strength) + target_y * flow_strength
        new_pos[node] = (x, new_y)
        
    return new_pos

# 1. 载入原始图数据并生成初始布局
file_name = 'grafo4323.78.graphml'  # 在这里切换数据集
file_path = os.path.join(DATASET_DIR, file_name)

if file_name.endswith('.mtx'): G = read_mtx(file_path)
elif file_name.endswith('.graphml'): G = read_graphml(file_path)

G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

print("正在计算初始布局...")
seed = stable_seed_from_name(file_name)
init_pos = apply_fa2(G, seed=seed)

# 2. 生成初始Cytoscape元素
elements = []
for node in G.nodes():
    x, y = init_pos[node]
    elements.append({
        'data': {'id': str(node), 'label': str(node), 'color': '#4287f5'},
        'position': {'x': float(x * 100), 'y': float(-y * 100)},
        'classes': 'node-cls'
    })
for edge in G.edges():
    elements.append({'data': {'source': str(edge[0]), 'target': str(edge[1])}})

# 3. 初始化 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2(f"交互式图布局调整 - {file_name}"),
    
    html.Div([
        html.Label("自上而下流向强度 (0.0=原始, 1.0=严格分层):"),
        dcc.Slider(
            id='flow-strength-slider',
            min=0.0, max=1.0, step=0.1, value=0.5,
            marks={i/10: str(i/10) for i in range(0, 11)}
        )
    ], style={'padding': '20px', 'width': '50%'}),
    
    html.Button("保存图片", id="save-btn", n_clicks=0, style={'marginBottom': '10px', 'padding': '10px'}),
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
                    'background-color': 'data(color)',
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

# 4. 回调函数：动态调整流向 & 保存图
@app.callback(
    [Output('cytoscape-graph', 'elements'),
     Output("output-msg", "children")],
    [Input('flow-strength-slider', 'value'),
     Input("save-btn", "n_clicks")],
    [State('cytoscape-graph', 'elements')]
)
def update_layout(flow_strength, n_clicks, current_elements):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'flow-strength-slider'
    
    # 如果点击保存图片
    if triggered_id == 'save-btn' and n_clicks > 0:
        new_pos = {}
        node_order = []
        
        for ele in current_elements:
            if 'position' in ele:
                node_id = ele['data']['id']
                n_obj = int(node_id) if node_id.isdigit() else node_id
                new_pos[n_obj] = (ele['position']['x'], -ele['position']['y'])
                node_order.append(n_obj)
        
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        
        nx.draw_networkx_nodes(G, new_pos, nodelist=node_order, node_size=20, node_color='#4287f5', alpha=0.9, edgecolors='black', linewidths=0.2)
        nx.draw_networkx_edges(G, new_pos, alpha=0.15, width=0.5)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{file_name}_flow{flow_strength}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dash.no_update, f"图片已成功保存至: {out_path}"

    # 如果滑动了流向滑块，计算新坐标
    flow_pos = apply_top_down_flow(G, init_pos, flow_strength=flow_strength)
    
    updated_elements = []
    for ele in current_elements:
        new_ele = ele.copy()
        if 'position' in new_ele:
            node_id = new_ele['data']['id']
            n_obj = int(node_id) if node_id.isdigit() else node_id
            
            x, y = flow_pos[n_obj]
            new_ele['position'] = {
                'x': float(x * 100),
                'y': float(-y * 100)
            }
        updated_elements.append(new_ele)
        
    return updated_elements, f"当前流向强度: {flow_strength}"

if __name__ == '__main__':
    print("启动 Dash 布局交互应用: http://127.0.0.1:8050")
    app.run(debug=True)