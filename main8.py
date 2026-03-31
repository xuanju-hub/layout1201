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
        outboundAttractionDistribution=False, linLogMode=False, adjustSizes=False,
        edgeWeightInfluence=1.0, jitterTolerance=1.0, barnesHutOptimize=True,
        barnesHutTheta=1.2, multiThreaded=False, scalingRatio=0.5,
        strongGravityMode=False, gravity=5.0
    )
    return forceatlas2.forceatlas2_networkx_layout(G, pos=init_pos, iterations=100)

def stable_seed_from_name(name: str, base: int = 2026) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base) % (2**31 - 1)

def apply_grid_snap(pos, grid_size=0.5, snap_strength=1.0):
    """将节点坐标向网格对齐"""
    if snap_strength == 0:
        return pos
    
    new_pos = {}
    for node, (x, y) in pos.items():
        # 计算最近的网格点
        grid_x = round(x / grid_size) * grid_size
        grid_y = round(y / grid_size) * grid_size
        
        # 混合原始坐标与网格坐标
        new_x = x * (1 - snap_strength) + grid_x * snap_strength
        new_y = y * (1 - snap_strength) + grid_y * snap_strength
        new_pos[node] = (new_x, new_y)
    return new_pos

def draw_orthogonal_edges(G, pos, ax):
    """在 matplotlib 中绘制 90 度正交边"""
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # 画出直角线 (x1,y1) -> (x1,y2) -> (x2,y2) 或者 (x1,y1) -> (x2,y1) -> (x2,y2)
        # 这里默认采用先水平后垂直的方式
        ax.plot([x1, x2, x2], [y1, y1, y2], color='#999999', alpha=0.3, linewidth=0.5, zorder=1)

# 1. 载入原始图数据并生成初始布局
file_name = 'grafo10444.100.graphml'
file_path = os.path.join(DATASET_DIR, file_name)

if file_name.endswith('.mtx'): G = read_mtx(file_path)
elif file_name.endswith('.graphml'): G = read_graphml(file_path)

G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
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
    html.H2(f"交互式正交边缘与网格对齐 - {file_name}"),
    
    html.Div([
        html.Label("边缘连线样式:"),
        dcc.RadioItems(
            id='edge-style-radio',
            options=[
                {'label': '原始直线', 'value': 'straight'},
                {'label': '90度正交 (Taxi)', 'value': 'taxi'}
            ],
            value='taxi',
            inline=True,
            style={'marginBottom': '10px'}
        ),
        html.Label("节点网格对齐强度 (0.0=原始布局, 1.0=严格对齐网格):"),
        dcc.Slider(
            id='snap-strength-slider',
            min=0.0, max=1.0, step=0.1, value=0.8,
            marks={i/10: str(i/10) for i in range(0, 11)}
        )
    ], style={'padding': '20px', 'width': '50%'}),
    
    html.Button("保存图片", id="save-btn", n_clicks=0, style={'marginBottom': '10px', 'padding': '10px'}),
    html.Div(id="output-msg", style={'color': 'green', 'marginBottom': '10px'}),
    
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        layout={'name': 'preset'}, 
        style={'width': '100%', 'height': '800px', 'border': '1px solid #ccc'}
    )
])

# 4. 回调函数：动态调整网格、改变边样式 & 保存图
@app.callback(
    [Output('cytoscape-graph', 'elements'),
     Output('cytoscape-graph', 'stylesheet'),
     Output("output-msg", "children")],
    [Input('edge-style-radio', 'value'),
     Input('snap-strength-slider', 'value'),
     Input("save-btn", "n_clicks")],
    [State('cytoscape-graph', 'elements')]
)
def update_layout(edge_style, snap_strength, n_clicks, current_elements):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'snap-strength-slider'
    
    # 动态设定 Cytoscape 边缘样式
    stylesheet = [
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
                'width': 1, 'line-color': '#999', 'opacity': 0.5,
                'curve-style': edge_style,
                'taxi-direction': 'auto' # Cytoscape 专用于正交的配置
            }
        }
    ]
    
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
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        
        # 画节点
        nx.draw_networkx_nodes(G, new_pos, nodelist=node_order, node_size=30, node_color='#4287f5', alpha=0.9, edgecolors='black', linewidths=0.2, ax=ax)
        
        # 画边
        if edge_style == 'taxi':
            draw_orthogonal_edges(G, new_pos, ax)
        else:
            nx.draw_networkx_edges(G, new_pos, alpha=0.15, width=0.5, ax=ax)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{file_name}_ortho_{edge_style}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dash.no_update, dash.no_update, f"图片已成功保存至: {out_path}"

    # 计算对齐网格后的新坐标
    # 根据图的一般尺寸，推测合适的网格大小。0.5 在FA2布局中通常间隔适中
    grid_pos = apply_grid_snap(init_pos, grid_size=0.5, snap_strength=snap_strength)
    
    updated_elements = []
    for ele in current_elements:
        new_ele = ele.copy()
        if 'position' in new_ele:
            node_id = new_ele['data']['id']
            n_obj = int(node_id) if node_id.isdigit() else node_id
            
            x, y = grid_pos[n_obj]
            new_ele['position'] = {
                'x': float(x * 100),
                'y': float(-y * 100)
            }
        updated_elements.append(new_ele)
        
    return updated_elements, stylesheet, f"当前网格对齐强度: {snap_strength}, 边样式: {edge_style}"

if __name__ == '__main__':
    print("启动 Dash 正交布局交互应用: http://127.0.0.1:8050")
    app.run(debug=True)