import numpy as np

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

def tianqin_positions_ssb_ecliptic_from_L(
    L, t_years=None, periods=1.0, N=2000,
    ecl_lon=0.0, ecl_lat=0.0, gamma=0.0, lambda0=0.0,
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
    if t_years is None:
        t_years_arr = np.linspace(0.0, periods*(T_sat/T_SID), N, endpoint=False)
    else:
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

# def tianqin_positions_ssb_ecliptic_from_L_years(
#     L, t_years=None, *, periods=1.0, N=2000,
#     ecl_lon=0.0, ecl_lat=0.0, gamma=0.0, lambda0=0.0,
#     earth_lambda0=0.0, a_AU=AU
# ):
#     """
#     L           : 臂长 [m]
#     t_years     : 时间数组（单位：年，恒星年）
#                   若为 None，则生成覆盖 periods 个天琴轨道周期（默认一年），N 点
#     ecl_lon     : 黄经 λ [rad]
#     ecl_lat     : 黄纬 β [rad]
#     gamma       : 平面内钟向角 [rad]
#     lambda0     : 三角形沿轨道的初相 [rad]
#     earth_lambda0: 地球在 t=0 的黄经相位 [rad]
#     a_AU        : 地球轨道半径 [m]
#     """
#     # 臂长 -> 三角形外接圆半径 R
#     R = L / np.sqrt(3.0)
#     omega = np.sqrt(MU_EARTH / R**3)  # 地心近似角速度 [rad/s]
#     T_sat = 2.0 * np.pi / omega       # 卫星地心周期 [s]

#     # 时间轴（年 -> 秒）
#     if t_years is None:
#         t_years_arr = np.linspace(0.0, periods*(T_sat/T_SID), N, endpoint=False)
#     else:
#         t_years_arr = np.atleast_1d(t_years)
#     t_sec = t_years_arr * T_SID  # 换算成秒

#     # 三星相位（彼此 120°）
#     lam = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
#     ph = lambda0 + omega * t_sec
#     c, s = np.cos(ph[None,:] + lam[:,None]), np.sin(ph[None,:] + lam[:,None])

#     # 局部（未旋转）构型
#     r_loc = np.zeros((3,3,len(t_sec)))
#     r_loc[:,0,:] = R * c
#     r_loc[:,1,:] = R * s

#     # 把局部 z 轴旋到黄道方向 (λ,β)，再绕 z 旋 gamma
#     Q = Rz(ecl_lon) @ Ry(np.pi/2 - ecl_lat) @ Rz(gamma)
#     r_constellation = np.einsum('ij,kjt->kit', Q, r_loc)

#     # 地球 SSB 黄道系圆轨道
#     rE = earth_circular_ssb_ecliptic(t_sec, a=a_AU, lambda0=earth_lambda0)

#     # SSB 黄道系三卫星位置
#     R_ssb = r_constellation + rE[None,:,:]
#     return R_ssb, rE, T_sat/T_SID, omega, t_years_arr  # 周期返回为年

# ------- 可选：臂向量 & 等效探测器张量（仍在 SSB 黄道系中） -------
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