# crcbpso.py
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union

Array = np.ndarray
FitRet = Union[Array, Tuple[Array, object, Array]]  # 兼容 “特殊边界条件” 返回 (F, _, newX)

def crcbpso(
    fitfunc_handle: Callable[[Array], FitRet],
    n_dim: int,
    p: Optional[Dict] = None,
    output_level: int = 0,
    seed_matrix: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Local-best (lbest) PSO minimizer in standardized coordinates.
    逐行对齐 MATLAB 版 crcbpso.m 的行为。

    Parameters
    ----------
    fitfunc_handle : function
        接收形状 (popSize, n_dim) 的标准化坐标 X ∈ [0,1]^n_dim，
        返回 fitness 向量 F（越小越好）。若启用“特殊边界条件”，
        可返回 (F, _, newX) 三元组，其中 newX 为可能被纠正后的坐标。
    n_dim : int
        搜索空间维数。
    p : dict, optional
        覆盖 PSO 参数。可用键：
        'popSize','maxSteps','c1','c2','maxVelocity',
        'startInertia','endInertia','endInertiaIter',
        'boundaryCond','nbrhdSz'
    output_level : int, optional
        0: 仅主输出；1: 另返回 allBestFit；2: 另返回 allBestLoc
    seed_matrix : ndarray, optional
        形状 (nSeed, nSeedCols)。将写入最前面的 nSeed 个粒子的前 nSeedCols 维。
       （与 MATLAB 行为一致）
    rng : np.random.Generator, optional
        随机数发生器

    Returns
    -------
    dict with keys:
        'totalFuncEvals', 'bestLocation', 'bestFitness',
        (可选) 'allBestFit', 'allBestLoc'
    """
    # ---------- 默认参数（与 MATLAB baseline 保持一致） ----------
    popsize       = 40
    max_steps     = 2000
    c1            = 2.0
    c2            = 2.0
    max_velocity  = 0.5
    start_inertia = 0.9
    end_inertia   = 0.4
    end_iter      = max_steps
    bndry_cond    = ''          # '' -> invisible wall
    nbrhd_sz      = 3

    if p:
        popsize       = p.get('popSize',       popsize)
        max_steps     = p.get('maxSteps',      max_steps)
        c1            = p.get('c1',            c1)
        c2            = p.get('c2',            c2)
        max_velocity  = p.get('maxVelocity',   max_velocity)
        start_inertia = p.get('startInertia',  start_inertia)
        end_inertia   = p.get('endInertia',    end_inertia)
        end_iter      = p.get('endInertiaIter',end_iter if 'maxSteps' not in p else p['maxSteps'])
        bndry_cond    = p.get('boundaryCond',  bndry_cond)
        nbrhd_sz      = max(int(p.get('nbrhdSz', nbrhd_sz)), 3)

    if rng is None:
        rng = np.random.default_rng()

    # ---------- 列索引布局（完全仿照 MATLAB） ----------
    partCoordCols      = slice(0, n_dim)                          # 位置
    partVelCols        = slice(n_dim, 2*n_dim)                    # 速度
    partPbestCols      = slice(2*n_dim, 3*n_dim)                  # 个体最好位置
    iFitPbest          = 3*n_dim                                  # pbest 的适应度
    iFitCurr           = iFitPbest + 1                            # 当前适应度
    iFitLbest          = iFitCurr + 1                             # 邻域最好适应度
    iInertia           = iFitLbest + 1                            # 当前粒子惯性权重
    partLocalBestCols  = slice(iInertia + 1, iInertia + 1 + n_dim)  # 邻域最好位置
    iFlagEval          = partLocalBestCols.stop                   # 是否计算适应度（0/1）
    iFitEvals          = iFlagEval + 1                            # 累计适评次数
    nColsPop           = iFitEvals + 1

    # ---------- 初始化种群 ----------
    pop = np.zeros((popsize, nColsPop), dtype=float)
    pop[:, partCoordCols] = rng.random((popsize, n_dim))
    # 种子坐标（行优先，列不超过 n_dim）
    if seed_matrix is not None and seed_matrix.size > 0:
        r, c = seed_matrix.shape
        r = min(r, popsize); c = min(c, n_dim)
        pop[:r, :c] = seed_matrix[:r, :c]

    pop[:, partVelCols]       = -pop[:, partCoordCols] + rng.random((popsize, n_dim))
    pop[:, partPbestCols]     = pop[:, partCoordCols]
    pop[:, iFitPbest]         = np.inf
    pop[:, iFitCurr]          = 0.0
    pop[:, iFitLbest]         = np.inf
    pop[:, iInertia]          = 0.0
    pop[:, partLocalBestCols] = 0.0
    pop[:, iFlagEval]         = 1.0
    pop[:, iFitEvals]         = 0.0

    # 全局最好
    gbest_val = np.inf
    gbest_loc = np.full(n_dim, 2.0)  # 与 MATLAB 一致：初始化为 2 (>1)

    # 邻域（环形拓扑）左右邻居个数
    left_nbrs  = (nbrhd_sz - 1) // 2
    right_nbrs = nbrhd_sz - 1 - left_nbrs

    # 扩展输出
    out = {}
    if output_level >= 1:
        out['allBestFit'] = np.zeros(max_steps)
    if output_level >= 2:
        out['allBestLoc'] = np.zeros((max_steps, n_dim))

    # ---------- 迭代 ----------
    for step in range(1, max_steps + 1):
        X = pop[:, partCoordCols]

        # 计算适应度（支持“特殊边界条件”三元返回）
        ret = fitfunc_handle(X) if bndry_cond == '' else fitfunc_handle(X)
        if isinstance(ret, tuple) and len(ret) == 3:
            fitness_values, _, newX = ret
            pop[:, partCoordCols] = newX
        else:
            fitness_values = ret  # 仅有 F

        # 填充当前适应度、pbest、计数
        for k in range(popsize):
            pop[k, iFitCurr] = fitness_values[k]
            compute_ok = (pop[k, iFlagEval] == 1.0)
            func_count = 1.0 if compute_ok else 0.0
            pop[k, iFitEvals] += func_count
            if pop[k, iFitPbest] > pop[k, iFitCurr]:
                pop[k, iFitPbest] = pop[k, iFitCurr]
                pop[k, partPbestCols] = pop[k, partCoordCols]

        # 更新全局最好（采用当前迭代的值）
        best_particle = int(np.argmin(pop[:, iFitCurr]))
        best_fitness  = float(pop[best_particle, iFitCurr])
        if gbest_val > best_fitness:
            gbest_val = best_fitness
            gbest_loc = pop[best_particle, partCoordCols].copy()
            # MATLAB 里这里还对 best 粒子额外加了一次 funcCount（略显奇怪），这里忽略

        # 本地最好（环形邻域）
        for k in range(popsize):
            ring = np.arange(k - left_nbrs, k + right_nbrs + 1)
            ring %= popsize
            lbest_idx = ring[np.argmin(pop[ring, iFitCurr])]
            lbest_val = pop[lbest_idx, iFitCurr]
            if lbest_val < pop[k, iFitLbest]:
                pop[k, iFitLbest] = lbest_val
                pop[k, partLocalBestCols] = pop[lbest_idx, partCoordCols]

        # 惯性权重线性衰减
        if end_iter <= 1:
            inertia_wt = end_inertia
        else:
            inertia_wt = max(
                start_inertia - ((start_inertia - end_inertia) / (end_iter - 1)) * (step - 1),
                end_inertia,
            )

        # 速度/位置更新 + 边界处理
        for k in range(popsize):
            pop[k, iInertia] = inertia_wt
            r1 = rng.random(n_dim)
            r2 = rng.random(n_dim)
            vel = (
                inertia_wt * pop[k, partVelCols]
                + c1 * r1 * (pop[k, partPbestCols] - pop[k, partCoordCols])
                + c2 * r2 * (pop[k, partLocalBestCols] - pop[k, partCoordCols])
            )
            # 速度裁剪（逐维）
            vel = np.clip(vel, -max_velocity, max_velocity)
            pop[k, partVelCols] = vel
            # 位置更新
            pop[k, partCoordCols] = pop[k, partCoordCols] + vel
            # “不可见墙”边界：出界则标记不评估，当前适应度设为 inf
            if np.any((pop[k, partCoordCols] < 0.0) | (pop[k, partCoordCols] > 1.0)):
                pop[k, iFitCurr] = np.inf
                pop[k, iFlagEval] = 0.0
            else:
                pop[k, iFlagEval] = 1.0

        # 扩展输出
        if output_level >= 1:
            out['allBestFit'][step - 1] = gbest_val
        if output_level >= 2:
            out['allBestLoc'][step - 1, :] = gbest_loc

    # 主输出
    total_evals = int(np.sum(pop[:, iFitEvals]))
    ret = {
        "totalFuncEvals": total_evals,
        "bestLocation":   gbest_loc,   # 标准化坐标
        "bestFitness":    gbest_val,
    }
    ret.update(out)
    return ret

