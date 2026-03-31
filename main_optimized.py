import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread
import warnings
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# 尝试导入社区检测库
try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# 尝试导入tqdm用于进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class LayoutType(Enum):
    """布局类型枚举"""
    STRESS = "stress"  # 紧凑风格 (Kamada-Kawai)
    FORCE = "force"    # 舒展风格 (Spring)


@dataclass
class LayoutConfig:
    """布局配置参数"""
    # Kamada-Kawai相关
    kamada_kawai_max_nodes: int = 500

    # Spring Layout相关
    spring_k: float = 0.15
    spring_iterations: int = 50
    spring_seed: int = 42

    # 可视化参数
    node_size: int = 15
    edge_width: float = 0.6
    edge_alpha: float = 0.5
    dpi: int = 120

    # 性能相关
    max_nodes_per_file: int = 10000
    memory_warning_threshold: int = 5000

    # 图像布局
    figsize_width: float = 10
    figsize_height_per_file: float = 4


class GraphVisualizer:
    """图可视化器类"""

    def __init__(self, config: Optional[LayoutConfig] = None):
        self.config = config or LayoutConfig()
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_graph_size(self, n_nodes: int, filename: str) -> bool:
        """验证图大小是否合适"""
        if n_nodes > self.config.max_nodes_per_file:
            self.logger.warning(f"图 {filename} 节点数过多({n_nodes})，跳过处理")
            return False

        if n_nodes > self.config.memory_warning_threshold:
            self.logger.warning(f"图 {filename} 节点数较大({n_nodes})，可能消耗较多内存")

        return True

    def load_local_graph(self, filepath: str) -> Optional[nx.Graph]:
        """加载本地图文件"""
        filename = os.path.basename(filepath)
        self.logger.info(f"正在加载: {filename}")

        try:
            # 1. 读取文件
            G = self._read_graph_file(filepath)
            if G is None:
                return None

            # 2. 预处理
            G = self._preprocess_graph(G)

            # 3. 验证图大小
            if not self._validate_graph_size(G.number_of_nodes(), filename):
                return None

            self.logger.info(f"加载成功: 节点={G.number_of_nodes()}, 边={G.number_of_edges()}")
            return G

        except FileNotFoundError:
            self.logger.error(f"文件不存在: {filepath}")
        except PermissionError:
            self.logger.error(f"文件权限不足: {filepath}")
        except Exception as e:
            self.logger.error(f"加载失败 {filename}: {type(e).__name__}: {e}")

        return None

    def _read_graph_file(self, filepath: str) -> Optional[nx.Graph]:
        """根据文件格式读取图"""
        try:
            if filepath.endswith('.mtx'):
                matrix = mmread(filepath)
                return nx.from_scipy_sparse_array(matrix)
            elif filepath.endswith('.graphml'):
                G = nx.read_graphml(filepath)
                return nx.convert_node_labels_to_integers(G)
            else:
                self.logger.warning(f"不支持的文件格式: {filepath}")
                return None
        except Exception as e:
            self.logger.error(f"读取文件内容失败: {e}")
            return None

    def _preprocess_graph(self, G: nx.Graph) -> nx.Graph:
        """预处理图数据"""
        # 转换为无向图
        G = G.to_undirected()

        # 移除自环
        G.remove_edges_from(nx.selfloop_edges(G))

        # 提取最大连通子图
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            self.logger.info(f"提取最大连通子图: {len(largest_cc)}个节点")

        # 移除权重信息
        for u, v, d in G.edges(data=True):
            d.clear()

        return G

    def _compute_layout(self, G: nx.Graph, layout_type: LayoutType) -> Dict:
        """计算布局"""
        n_nodes = G.number_of_nodes()

        if layout_type == LayoutType.STRESS:
            if n_nodes > self.config.kamada_kawai_max_nodes:
                self.logger.info(f"节点数({n_nodes})过多，使用Spring布局替代Kamada-Kawai")
                return nx.spring_layout(
                    G,
                    k=1.0/((n_nodes)**0.5),
                    iterations=self.config.spring_iterations*2,
                    seed=self.config.spring_seed
                )
            else:
                return nx.kamada_kawai_layout(G)
        else:  # LayoutType.FORCE
            return nx.spring_layout(
                G,
                k=self.config.spring_k,
                iterations=self.config.spring_iterations,
                seed=self.config.spring_seed
            )

    def _compute_colors(self, G: nx.Graph) -> Tuple[List, List]:
        """计算节点和边的颜色"""
        if not HAS_LOUVAIN:
            return ['#4e79a7'] * G.number_of_nodes(), ['#b0b0b0'] * G.number_of_edges()

        # 社区检测
        partition = community_louvain.best_partition(G)

        # 颜色映射
        unique_comms = list(set(partition.values()))
        cmap = plt.cm.Spectral
        norm = plt.Normalize(vmin=min(unique_comms), vmax=max(unique_comms))

        # 节点颜色
        node_colors = [cmap(norm(partition[n])) for n in G.nodes()]

        # 边颜色
        edge_colors = []
        for u, v in G.edges():
            if partition[u] == partition[v]:
                edge_colors.append(cmap(norm(partition[u])))
            else:
                edge_colors.append('#cccccc')

        return node_colors, edge_colors

    def draw_graph(self, ax, G: nx.Graph, layout_type: LayoutType, title: str):
        """绘制图"""
        # 1. 计算布局
        t0 = time.time()
        pos = self._compute_layout(G, layout_type)
        layout_time = time.time() - t0

        # 2. 计算颜色
        node_colors, edge_colors = self._compute_colors(G)

        # 3. 绘制边
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=self.config.edge_width,
            alpha=self.config.edge_alpha,
            arrows=False
        )

        # 4. 绘制节点
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=self.config.node_size,
            node_color=node_colors,
            linewidths=0
        )

        # 5. 设置标题和样式
        ax.set_title(f"{title}\n({layout_time:.2f}s)", fontsize=10)
        ax.set_axis_off()

    def process_files(self, data_dir: str = "datasets"):
        """处理所有图文件"""
        # 1. 获取文件列表
        files = self._get_files(data_dir)

        if not files:
            self.logger.error(f"在 {data_dir} 文件夹中没有找到图文件")
            return

        # 2. 设置画布
        n_files = len(files)
        fig, axes = plt.subplots(
            n_files, 2,
            figsize=(self.config.figsize_width, self.config.figsize_height_per_file * n_files),
            dpi=self.config.dpi
        )

        if n_files == 1:
            axes = [axes]

        # 3. 处理每个文件
        self.logger.info(f"开始处理 {n_files} 个图文件")

        files_iter = tqdm(files, desc="处理图文件") if HAS_TQDM else files

        for i, filepath in enumerate(files_iter):
            filename = os.path.basename(filepath)

            # 加载图
            G = self.load_local_graph(filepath)
            if G is None:
                continue

            # 绘制两种布局
            self.logger.info(f"绘制 {filename} - Stress布局")
            self.draw_graph(
                axes[i][0], G, LayoutType.STRESS,
                f"{filename}\n(Stress/SGD2 Style)"
            )

            self.logger.info(f"绘制 {filename} - Force布局")
            self.draw_graph(
                axes[i][1], G, LayoutType.FORCE,
                f"{filename}\n(Force/SmartGD Style)"
            )

        # 4. 显示结果
        plt.tight_layout()
        plt.show()
        self.logger.info("处理完成")

    def _get_files(self, data_dir: str) -> List[str]:
        """获取所有图文件"""
        data_path = Path(data_dir)
        if not data_path.exists():
            self.logger.error(f"目录不存在: {data_dir}")
            return []

        # 支持的文件扩展名
        extensions = ['*.mtx', '*.graphml']
        files = []

        for ext in extensions:
            files.extend(glob.glob(os.path.join(data_dir, ext)))

        # 排序保证一致性
        files.sort()

        if not files:
            available_files = list(data_path.glob('*'))
            if available_files:
                self.logger.info(f"目录中的文件: {[f.name for f in available_files]}")

        return files


def main():
    """主函数"""
    # 配置参数
    config = LayoutConfig(
        kamada_kawai_max_nodes=300,  # 降低阈值以提高性能
        spring_iterations=100,        # 增加迭代次数提高质量
        spring_k=0.1,                # 调整k值
    )

    # 创建可视化器
    visualizer = GraphVisualizer(config)

    # 处理文件
    visualizer.process_files()


if __name__ == "__main__":
    # 忽略matplotlib的警告
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    main()