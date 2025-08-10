import numpy as np
from .ligo_antenna import rotate_by_psi

# 常量
AU     = 1.495978707e11             # [m]
T_SID  = 365.256363004 * 86400.0    # 恒星年 [s]

# 基本旋转
def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]])

# 地球在 SSB 黄道系的简化圆轨道（黄道平面）
def earth_circular_ssb_ecliptic(t, a=AU, lambda0=0.0):
    """
    t: (Nt,) 秒；返回 r_E^SSB (3,Nt) —— 黄道坐标系，Z=0。
    """
    t = np.atleast_1d(t)
    Omega = 2*np.pi / T_SID
    th = Omega * t + lambda0
    return np.vstack((a*np.cos(th), a*np.sin(th), np.zeros_like(th)))

# 天琴：在 SSB 黄道系中生成三卫星（只传入臂长 L）
MU_EARTH = 3.986004418e14  # [m^3/s^2]

def tianqin_positions(
    L, t_years=None, ecl_lon=0.0, ecl_lat=0.0, gamma=0.0, lambda0=0.0,
    earth_lambda0=0.0, a_AU=AU
):
    """
    直接在 SSB 黄道坐标系构造位置：
      - 三星局部圆轨道半径 R=L/√3，角速度 ω=√(μ/R^3)，周期 T=2π/ω（地心引力近似）
      - 平面法线由黄道经纬 (ecl_lon, ecl_lat) 决定；gamma 为平面内钟向角
      - SSB 位置 = 地球(黄道) + 三星(以地心构型旋到黄道系)

    参数：
      L           : 臂长 [m]
      t / periods,N: 时间轴（单位：年，恒星年）；t 为 None 时生成覆盖 periods 个周期、N 点
      ecl_lon     : 目标法线的**黄经** λ（rad）
      ecl_lat     : 目标法线的**黄纬** β（rad）
      gamma       : 平面内钟向角（rad）
      lambda0     : 三角形沿轨道的初相（rad）
      earth_lambda0: 地球在 t=0 的黄经相位（rad）
      a_AU        : 地球圆轨道半径（默认 1 AU）

    返回：
      R_ssb : (3,3,Nt) 三卫星在 SSB 黄道系的位置
      rE    : (3,Nt)   地球 SSB 黄道系位置
      T, omega, t_arr
    """
    # L -> R, ω, T
    R = L / np.sqrt(3.0)
    omega = np.sqrt(MU_EARTH / R**3)    # 地心引力近似
    T_sat = 2.0 * np.pi / omega

    # 时间轴（年 -> 秒）
    # if t_years is None:
    #     t_years_arr = np.linspace(0.0, periods*(T_sat/T_SID), N, endpoint=False)
    # else:
    t_years_arr = np.atleast_1d(t_years)
    t_arr = t_years_arr * T_SID  # 换算成秒
    Nt = len(t_arr)

    # 三星相位（彼此 120°）
    lam = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
    ph = lambda0 + omega * t_arr           # (Nt,)
    c, s = np.cos(ph[None,:] + lam[:,None]), np.sin(ph[None,:] + lam[:,None])

    # 局部（未旋转）构型：位于 x–y 平面
    r_loc = np.zeros((3,3,Nt))
    r_loc[:,0,:] = R * c
    r_loc[:,1,:] = R * s
    # z=0

    # 把局部 z 轴旋到黄道方向 (λ,β)，再绕 z 旋 gamma
    # 注：与赤道系的写法同形，但这里解释为“在黄道系内”的旋转
    Q = Rz(ecl_lon) @ Ry(np.pi/2 - ecl_lat) @ Rz(gamma)  # 3x3
    r_constellation = np.einsum('ij,kjt->kit', Q, r_loc) # (3,3,Nt)

    # 地球 SSB 黄道系圆轨道
    rE = earth_circular_ssb_ecliptic(t_arr, a=a_AU, lambda0=earth_lambda0)  # (3,Nt)

    # SSB 黄道系下的三卫星位置 = 地球 + 构型
    R_ssb = r_constellation + rE[None,:,:]
    # return R_ssb, rE, T_sat/T_SID, omega, t_years_arr 
    return R_ssb, rE

def tianqin_antenna_patterns_numerical(L, t_years, theta, phi,
    ecl_lon=0.0, ecl_lat=0.0, gamma=0.0, lambda0=0.0,
    earth_lambda0=0.0, a_AU=AU, psi=0.0, eps=1e-15):
    """
    计算全天 (theta, phi) 网格随时间的天线模式：
    输入:
        L           : 臂长 [m]
        t_years     : 时间轴（单位：年，恒星年）
        theta, phi  : 天球坐标（rad），可为标量或网格
        ecl_lon     : 目标法线的**黄经** λ（rad）
        ecl_lat     : 目标法线的**黄纬** β（rad）
        gamma       : 平面内钟向角（rad）
        lambda0     : 三角形沿轨道的初相（rad）
        earth_lambda0: 地球在 t=0 的黄经相位（rad）
        a_AU        : 地球圆轨道半径（默认 1 AU）
        psi         : 极化角（rad），可为标量或 (T,) 数组
        eps         : 数值稳定性小常数
    输出:
        F1_plus, F1_cross, F2_plus, F2_cross  形状均为 (Nθ, Nφ, T)
    """
    # 1) 轨道与臂单位向量 (3,T)
    R_ssb, _ = tianqin_positions(
        L=L, t_years=t_years, ecl_lon=ecl_lon, ecl_lat=ecl_lat,
        gamma=gamma, lambda0=lambda0, earth_lambda0=earth_lambda0, a_AU=a_AU
    )
    orb1, orb2, orb3 = R_ssb[0], R_ssb[1], R_ssb[2]  # (3,Nt)
    arm1 = orb1 - orb2
    arm2 = orb2 - orb3
    arm3 = orb3 - orb1
    arm1 /= np.maximum(np.linalg.norm(arm1, axis=0, keepdims=True), eps)
    arm2 /= np.maximum(np.linalg.norm(arm2, axis=0, keepdims=True), eps)
    arm3 /= np.maximum(np.linalg.norm(arm3, axis=0, keepdims=True), eps)

    # 2) 两个等效探测器张量 D^I, D^II (3,3,T)
    o11 = np.einsum('it,jt->ijt', arm1, arm1)
    o22 = np.einsum('it,jt->ijt', arm2, arm2)
    o33 = np.einsum('it,jt->ijt', arm3, arm3)
    D1  = 0.5 * (o11 - o22)
    D2  = (1.0/(2.0*np.sqrt(3.0))) * (o11 + o22 - 2.0*o33)

    # 3) 广播 theta/phi 到同形状（既兼容标量也兼容网格）
    TH, PH = np.broadcast_arrays(theta, phi)

    # 4) 全天/单点方向基 (...,3)
    direction = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ], axis=-1)

    # # 5) 构造波参考基并规避极点奇异
    # ref_z = np.array([0.0, 0.0, 1.0])
    # ref_x = np.array([1.0, 0.0, 0.0])
    # use_x = (np.abs(direction[..., 2]) > 0.9)[..., None]          # (...,1)
    # # 自动广播到 (...,3)
    # ref = np.where(use_x, ref_x, ref_z)

    # x_wave = np.cross(ref, direction)
    # x_wave /= np.maximum(np.linalg.norm(x_wave, axis=-1, keepdims=True), eps)
    # y_wave = np.cross(direction, x_wave)
    # y_wave /= np.maximum(np.linalg.norm(y_wave, axis=-1, keepdims=True), eps)
    # 5) 直接用球面正交基，避免极点切换带来的不连续
    x_wave = np.stack([np.cos(TH)*np.cos(PH),
                    np.cos(TH)*np.sin(PH),
                    -np.sin(TH)], axis=-1)        # ˆe_theta
    y_wave = np.stack([-np.sin(PH),
                    np.cos(PH),
                    np.zeros_like(TH)], axis=-1)   # ˆe_phi

    # 6) 偏振张量 e_plus/e_cross 形状 (...,3,3)
    e_plus  = x_wave[..., :, None] * x_wave[..., None, :] - y_wave[..., :, None] * y_wave[..., None, :]
    e_cross = x_wave[..., :, None] * y_wave[..., None, :] + y_wave[..., :, None] * x_wave[..., None, :]

    # 7) 与探测器缩并 -> (...,T)
    F1p = np.einsum('...ij,ijt->...t', e_plus,  D1)
    F1x = np.einsum('...ij,ijt->...t', e_cross, D1)
    F2p = np.einsum('...ij,ijt->...t', e_plus,  D2)
    F2x = np.einsum('...ij,ijt->...t', e_cross, D2)

    # 8) 极化角旋转（psi 可为标量或 (T,)）
    F1p, F1x = rotate_by_psi(F1p, F1x, psi)
    F2p, F2x = rotate_by_psi(F2p, F2x, psi)
    return F1p, F1x, F2p, F2x

def arm_vectors(R):
    r1, r2, r3 = R[0], R[1], R[2]   # (3,Nt)
    n12 = r2 - r1
    n23 = r3 - r2
    n31 = r1 - r3
    return n12, n23, n31

def unit(v, eps=1e-15):
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return v / np.maximum(n, eps)

def detector_tensors_DI_DII_from_arms(n12, n23, n31):
    u = unit(n12)
    v = unit(-n31)
    w = unit(n23)
    outer = lambda a,b: np.einsum('it,jt->ijt', a,b)
    DI  = 0.5 * (outer(u,u) - outer(v,v))
    DII = (1.0/(2.0*np.sqrt(3.0))) * (outer(u,u) + outer(v,v) - 2.0*outer(w,w))
    return DI, DII

def equatorial_to_ecliptic(ra, dec, obliquity_deg=23.439281):
    """
    ra, dec: 赤经/赤纬 (rad)，可标量或数组
    obliquity_deg: 黄赤交角 (deg)，默认 J2000 值
    返回: (lambda_ecl, beta_ecl) in rad
    """
    ra   = np.asarray(ra, dtype=float)
    dec  = np.asarray(dec, dtype=float)
    eps  = np.deg2rad(obliquity_deg)

    cosd, sind = np.cos(dec), np.sin(dec)
    cosa, sina = np.cos(ra),  np.sin(ra)

    x_eq = cosd * cosa
    y_eq = cosd * sina
    z_eq = sind

    # rotate about +x by -eps : (x stays), y', z'
    y_ecl = y_eq * np.cos(eps) + z_eq * np.sin(eps)
    z_ecl = -y_eq * np.sin(eps) + z_eq * np.cos(eps)
    x_ecl = x_eq

    lam = np.arctan2(y_ecl, x_ecl)            # [-pi, pi]
    lam = np.mod(lam, 2.0*np.pi)              # [0, 2pi)
    beta = np.arcsin(np.clip(z_ecl, -1.0, 1.0))
    return lam, beta