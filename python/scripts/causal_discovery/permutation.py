"""BOSS 与 GRaSP 算法封装 —— 基于 causal-learn 库。

两种算法均属于**基于排列的打分搜索（Permutation-based Score Search）**范式，
通过在变量排列（causal order）空间中搜索最优分数来学习因果结构。

算法简介
--------

BOSS (Better Order Score Search, Lam et al., 2022)
    在变量排列空间中迭代执行"更优突变"（Better Mutation），
    每次对单个变量尝试所有可能的插入位置，保留使总分提升的移动，
    直到不再有提升为止。最终由最优排列推导出 DAG，再转换为 CPDAG。

GRaSP (Greedy Relaxations of the Sparsest Permutation, Desmarais et al., 2022)
    从最稀疏排列（Sparsest Permutation）问题出发，
    通过有限深度 DFS 贪心地松弛排列约束，寻找更优的排列，
    最终输出 CPDAG。depth 参数控制 DFS 深度（默认 3），
    越大越准但越慢。

与 GES 的比较
-------------
- GES 在等价类空间（CPDAG 空间）中搜索；BOSS/GRaSP 在排列空间中搜索，
  等价于在 DAG 空间搜索后再转 CPDAG。
- 理论上 BOSS/GRaSP 在有限样本下更鲁棒（因为排列空间结构更简单），
  可作为 GES 的鲁棒替代或交叉验证补充。

评分函数选择
------------
两种算法均支持相同的评分函数集合：
  - **auto**：离散数据 → ``local_score_BDeu``；连续数据 → ``local_score_BIC_from_cov``。
  - **local_score_BDeu**：贝叶斯 Dirichlet 评分，适用于离散数据。
  - **local_score_BIC_from_cov**：线性高斯 BIC（基于协方差），适用于连续数据，速度快。
  - **local_score_BIC**：线性高斯 BIC（原始版），内部自动重定向至 BIC_from_cov。

图格式约定（同 GES）
--------------------
causal-learn 返回 GeneralGraph，约定：
  G.graph[j,i] = 1, G.graph[i,j] = -1  →  i → j
  G.graph[i,j] = G.graph[j,i] = -1     →  无向边 i — j（CPDAG 中未定向）

本项目统一邻接矩阵约定：
  adj[i,j] = 1  表示存在 i→j 的有向边或 i—j 的无向边。
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from .base import BaseDiscoveryMethod
from .data_utils import detect_data_type


def _resolve_score_func(score_func: str, data_array: np.ndarray) -> str:
    """将 "auto" 解析为具体的评分函数名称。

    离散数据使用 BDeu（贝叶斯 Dirichlet 均匀先验），
    连续数据使用 BIC_from_cov（线性高斯 BIC，协方差版）。
    """
    if score_func != "auto":
        return score_func
    dtype = detect_data_type(data_array)
    return "local_score_BDeu" if dtype == "discrete" else "local_score_BIC_from_cov"


def _convert_graph(raw: np.ndarray) -> np.ndarray:
    """将 causal-learn GeneralGraph 格式转换为项目邻接矩阵约定。

    causal-learn 约定:
        raw[j,i] = 1, raw[i,j] = -1  →  i → j
        raw[i,j] = raw[j,i] = -1     →  无向边 i — j

    项目约定:
        adj[i,j] = 1  →  存在 i→j 或 i—j 的边
    """
    # i → j: raw[i,j] == -1 且 raw[j,i] == 1
    directed = (raw == -1) & (raw.T == 1)
    # 无向边: raw[i,j] == -1 且 raw[j,i] == -1
    undirected = (raw == -1) & (raw.T == -1)

    adj = np.zeros(raw.shape, dtype=int)
    adj[directed] = 1
    adj[undirected] = 1
    return adj


class BOSSAlgorithm(BaseDiscoveryMethod):
    """BOSS (Better Order Score Search, Lam et al., 2022).

    适用场景
    --------
    - 连续或离散数据均支持（通过 ``score_func`` 配置）。
    - 对于 alarm 等离散贝叶斯网络数据集，推荐 ``score_func = "auto"``
      或 ``"local_score_BDeu"``。

    算法特点
    --------
    - 基于排列搜索，不在等价类空间搜索，减少了局部极值的风险。
    - 实践中与 GES 速度相当。
    - 输出 CPDAG（由最优 DAG 经 dag2cpdag 转换）。
    """

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """运行 BOSS 算法。

        Parameters
        ----------
        data : pd.DataFrame
            形状为 (N, D) 的观测数据。
        score_func : str
            评分函数；"auto" 表示根据数据类型自动选择。

        Returns
        -------
        np.ndarray
            形状为 (D, D) 的 CPDAG 邻接矩阵。
        """
        from causallearn.search.PermutationBased.BOSS import boss

        score_func = kwargs.get("score_func", "auto")

        data_array: np.ndarray = data.to_numpy()
        score_func = _resolve_score_func(score_func, data_array)

        node_names = list(data.columns)

        # verbose=False 是必须的：默认 True 会往 stdout 打印，破坏 JSON 解析
        G = boss(
            data_array,
            score_func=score_func,
            verbose=False,
            node_names=node_names,
        )

        return _convert_graph(G.graph)


class GRaSPAlgorithm(BaseDiscoveryMethod):
    """GRaSP (Greedy Relaxations of the Sparsest Permutation, Desmarais et al., 2022).

    适用场景
    --------
    - 连续或离散数据均支持（通过 ``score_func`` 配置）。
    - 对于 alarm 等离散贝叶斯网络数据集，推荐 ``score_func = "auto"``
      或 ``"local_score_BDeu"``。

    算法特点
    --------
    - 基于稀疏排列松弛的贪心搜索，通过有限深度 DFS 探索排列空间。
    - ``depth`` 越大，搜索越彻底，但运行时间随之增长；默认 depth=3。
    - 输出 CPDAG（由最优 DAG 经 dag2cpdag 转换）。
    """

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """运行 GRaSP 算法。

        Parameters
        ----------
        data : pd.DataFrame
            形状为 (N, D) 的观测数据。
        score_func : str
            评分函数；"auto" 表示根据数据类型自动选择。
        depth : int
            DFS 搜索深度，越大越准但越慢，默认 3。

        Returns
        -------
        np.ndarray
            形状为 (D, D) 的 CPDAG 邻接矩阵。
        """
        from causallearn.search.PermutationBased.GRaSP import grasp

        score_func = kwargs.get("score_func", "auto")
        depth = kwargs.get("depth", 3)

        data_array: np.ndarray = data.to_numpy()
        score_func = _resolve_score_func(score_func, data_array)

        node_names = list(data.columns)

        # verbose=False 是必须的：默认 True 会往 stdout 打印，破坏 JSON 解析
        G = grasp(
            data_array,
            score_func=score_func,
            depth=depth,
            verbose=False,
            node_names=node_names,
        )

        return _convert_graph(G.graph)
