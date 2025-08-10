# test_crcbpso.py
import time
import numpy as np
import matplotlib.pyplot as plt

from crcbpso import crcbpso
from crcbpsotestfunc import crcbpsotestfunc, r2sv  # 只用到 crcbpsotestfunc

# ==== 参数 ====
nDim = 20
rmin = -10.0
rmax =  10.0
ffparams = {"rmin": rmin, "rmax": rmax}

# fitness 函数句柄（标准化坐标 -> 标准 Rastrigin）
fitFuncHandle = lambda X: crcbpsotestfunc(X, ffparams)

# ---- 打印“默认 PSO 设置”（与 MATLAB baseline 对齐）----
def pso_defaults():
    return {
        "popSize":       40,
        "maxSteps":      2000,
        "c1":            2.0,
        "c2":            2.0,
        "maxVelocity":   0.5,
        "startInertia":  0.9,
        "endInertia":    0.4,
        "endInertiaIter":2000,
        "boundaryCond":  "",
        "nbrhdSz":       3,
    }

print("Default PSO settings")
print(pso_defaults())

# ==== 以默认设置调用（不返回额外信息）====
print("Calling PSO with default settings and no optional inputs")
rng = np.random.default_rng(0)   # 等价 rng('default')
t0 = time.time()
psoOut1 = crcbpso(fitFuncHandle, nDim, rng=rng)  # output_level=0
t1 = time.time()
print(f"Elapsed: {t1-t0:.3f}s")

# ==== 以默认设置调用（返回扩展信息 allBestFit/allBestLoc）====
print("Calling PSO with default settings and optional inputs")
rng = np.random.default_rng(0)   # 重置随机种子
t0 = time.time()
psoOut1 = crcbpso(fitFuncHandle, nDim, output_level=2, rng=rng)
t1 = time.time()
print(f"Elapsed: {t1-t0:.3f}s")

# 最优标准化坐标 -> 实坐标（调用 fitness 返回坐标的接口）
stdCoord = psoOut1["bestLocation"][None, :]                 # 变成 1xD
F, realCoord, _ = crcbpsotestfunc(stdCoord, ffparams, return_coords=True)
realCoord = realCoord[0]
print(" Best location:", np.array2string(realCoord, precision=4))
print(" Best fitness:",  psoOut1["bestFitness"])

# ==== 覆盖部分 PSO 参数 ====
print("Overriding default PSO parameters")
psoParams = {
    "maxSteps":    30000,
    "maxVelocity": 0.9,
}
print("Changing maxSteps   to:", psoParams["maxSteps"])
print("Changing maxVelocity to:", psoParams["maxVelocity"])
rng = np.random.default_rng(0)
t0 = time.time()
psoOut2 = crcbpso(fitFuncHandle, nDim, p=psoParams, output_level=2, rng=rng)
t1 = time.time()
print(f"Elapsed: {t1-t0:.3f}s")

# ==== 结果图 ====
plt.figure()
plt.plot(psoOut1["allBestFit"])
plt.xlabel("Iteration number")
plt.ylabel("Global best fitness")
plt.title("Default PSO settings")

plt.figure()
plt.plot(psoOut2["allBestFit"])
plt.xlabel("Iteration number")
plt.ylabel("Global best fitness")
plt.title("Non-default PSO settings")

# 若 nDim==2，画等高线 + 最优轨迹
if nDim == 2:
    # 网格（真实坐标）
    xGrid = np.linspace(rmin, rmax, 500)
    yGrid = np.linspace(rmin, rmax, 500)
    X, Y = np.meshgrid(xGrid, yGrid, indexing="xy")
    # 标准化坐标（与 MATLAB 一致）
    Xstd = (X - rmin) / (rmax - rmin)
    Ystd = (Y - rmin) / (rmax - rmin)
    fitVal4plot = fitFuncHandle(np.c_[Xstd.ravel(), Ystd.ravel()]).reshape(X.shape)

    # 在标准化坐标上画等高线 + 最优轨迹（PSO 返回的也是标准化）
    plt.figure()
    cs = plt.contour((xGrid - rmin)/(rmax - rmax + rmax - rmin),  # 仅做轴映射时可直接用 Xstd/Ystd
                     (yGrid - rmin)/(rmax - rmin),
                     fitVal4plot)
    plt.clabel(cs, inline=True, fontsize=8)
    loc = psoOut2["allBestLoc"]  # (iters, 2) 标准化
    plt.plot(loc[:,0], loc[:,1], ".-")
    plt.title("Trajectory of the best particle")

    # 画 3D 曲面（真实坐标）
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, fitVal4plot, linewidth=0, antialiased=True)
    ax.set_title("Plot of fitness function")

plt.show()

# 最后再打印一次非默认设置的最优点
stdCoord = psoOut2["bestLocation"][None, :]
F, realCoord, _ = crcbpsotestfunc(stdCoord, ffparams, return_coords=True)
realCoord = realCoord[0]
print(" Best location:", np.array2string(realCoord, precision=4))
print(" Best fitness:",  psoOut2["bestFitness"])

