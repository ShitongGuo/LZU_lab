# crcbqcpso.py
import numpy as np
from typing import Any, Dict, List

from crcbpso import crcbpso              # 你的 PSO 主函数（我们之前已经实现）
from crcbqcfitfunc import crcbqcfitfunc  # 适应度函数：必须与 MATLAB 版接口一致
from qc_signal import crcbgenqcsig1       # 若你使用 crcbgenqcsig1，请把这行换成 from qc_signal import crcbgenqcsig1 as crcbgenqcsig


def crcbqcpso(inParams: Dict[str, Any],
              psoParams: Dict[str, Any],
              nRuns: int) -> Dict[str, Any]:
    """
    Regression of quadratic chirp using PSO.
    严格按 MATLAB: crcbqcpso.m 实现。
    Inputs (inParams 必须含有下列字段，名称对齐 MATLAB):
      - dataY : (N,) 数据向量
      - dataX : (N,) 时间戳
      - dataXSq : (N,) dataX**2
      - dataXCb : (N,) dataX**3
      - rmin, rmax : (3,) 三个参数的搜索下/上界（供 fitfunc 用）
      - 其他 fitfunc 需要的字段（例如 psd、fs 等），保持在 inParams 内由 crcbqcfitfunc 使用

    psoParams: PSO 参数字典（键与 crcbpso 保持一致）
    nRuns: 独立 PSO 跑的次数

    Returns: outResults 字典（字段名、层级与 MATLAB 一致）
    """
    nSamples = len(inParams["dataX"])

    # fHandle = @(x) crcbqcfitfunc(x, inParams);
    def fHandle(X: np.ndarray):
        return crcbqcfitfunc(X, inParams)

    nDim = 3

    # outStruct 模板（对应 MATLAB 中每个 run 的 crcbpso 输出）
    outStruct: List[Dict[str, Any]] = [
        {"bestLocation": None, "bestFitness": None, "totalFuncEvals": None}
        for _ in range(nRuns)
    ]

    # 总输出（字段与 MATLAB 一致）
    outResults: Dict[str, Any] = {
        "allRunsOutput": [
            {
                "fitVal": None,
                "qcCoefs": np.zeros(3),
                "estSig": np.zeros(nSamples),
                "totalFuncEvals": None,
            }
            for _ in range(nRuns)
        ],
        "bestRun": None,
        "bestFitness": None,
        "bestSig": np.zeros(nSamples),
        "bestQcCoefs": np.zeros(3),
    }

    # parfor lpruns = 1:nRuns
    #   rng(lpruns); outStruct(lpruns)=crcbpso(fHandle,nDim,psoParams);
    for lpruns in range(1, nRuns + 1):
        rng = np.random.default_rng(lpruns)  # 与 MATLAB rng(lpruns) 对齐
        outStruct[lpruns - 1] = crcbpso(
            fHandle,
            nDim,
            p=psoParams,
            output_level=0,
            rng=rng,
        )

    # Prepare output: 填充每个 run 的 fitVal / qcCoefs / estSig / totalFuncEvals
    fitVal = np.zeros(nRuns)
    for i in range(nRuns):
        # best fitness
        fitVal[i] = outStruct[i]["bestFitness"]
        outResults["allRunsOutput"][i]["fitVal"] = fitVal[i]

        # [~, qcCoefs] = fHandle(outStruct(lpruns).bestLocation);
        # 这里要求 crcbqcfitfunc 在被单点（1×3 标准化坐标）调用时，返回 (F, qcCoefs)
        _, qcCoefs = crcbqcfitfunc(
            np.asarray(outStruct[i]["bestLocation"], dtype=float).reshape(1, -1),
            inParams,
        )
        qcCoefs = np.asarray(qcCoefs).ravel()
        outResults["allRunsOutput"][i]["qcCoefs"] = qcCoefs

        # estSig = crcbgenqcsig(inParams.dataX,1,qcCoefs);
        estSig_unit = crcbgenqcsig(inParams["dataX"], 1.0, qcCoefs)

        # estAmp = inParams.dataY*estSig(:); estSig = estAmp*estSig;
        estAmp = float(np.dot(inParams["dataY"].ravel(), estSig_unit.ravel()))
        estSig = estAmp * estSig_unit
        outResults["allRunsOutput"][i]["estSig"] = estSig

        # total func evals
        outResults["allRunsOutput"][i]["totalFuncEvals"] = outStruct[i]["totalFuncEvals"]

    # Find the best run
    bestRun_idx0 = int(np.argmin(fitVal))      # 0-based
    bestRun = bestRun_idx0 + 1                 # 与 MATLAB 一致：返回 1-based 下标
    outResults["bestRun"] = bestRun
    outResults["bestFitness"] = outResults["allRunsOutput"][bestRun_idx0]["fitVal"]
    outResults["bestSig"] = outResults["allRunsOutput"][bestRun_idx0]["estSig"]
    outResults["bestQcCoefs"] = outResults["allRunsOutput"][bestRun_idx0]["qcCoefs"]

    return outResults



