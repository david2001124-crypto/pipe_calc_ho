import streamlit as st
from CoolProp.CoolProp import PropsSI, PhaseSI
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
import material_db

# ==========================================
# 앱 기본 설정 및 세션 초기화
# ==========================================
st.set_page_config(page_title="해양 파이프라인 시뮬레이터", layout="wide")

# DB 호환성 안전장치: 구버전 캐시가 남아있을 경우 에러 방지
if hasattr(material_db, 'HYSYS_FITTING_DB'):
    FITTING_DB = material_db.HYSYS_FITTING_DB
else:
    FITTING_DB = {"Valve/Fitting (Default)": {"A": 0.0, "B": 30, "Chisholm_B": 1.5}}

# ==========================================
# 1. 공학/물리 계산 헬퍼 함수 모음
# ==========================================

def churchill_friction_factor(Re, ed):
    """
    [Churchill (1977) 마찰계수 방정식]
    층류, 전이 영역, 난류 전 구간을 불연속점 없이 하나의 수식으로 계산.
    반복 해석 시 발산(터짐)을 막아줍니다.
    """
    if Re <= 0: return 0.01 
    A = (-2.457 * math.log((7.0 / Re)**0.9 + 0.27 * ed))**16
    B = (37530.0 / Re)**16
    f = 8 * ((8 / Re)**12 + 1 / (A + B)**1.5)**(1/12)
    return f

def calculate_fT_hysys(roughness, D_inner):
    """
    [완전 난류 마찰계수(fT) 계산 - HYSYS/Crane TP-410 방식]
    유속을 무한대(Re -> 무한대)로 가속시켰을 때 마찰계수가 더 이상 변하지 않고
    배관 거칠기에 의해서만 결정되는 한계 수렴 지점을 반복 호출로 찾아냅니다.
    """
    Re_test = 1e6 # 초기 레이놀즈 수를 난류로 크게 설정
    f_old = 0.0
    for _ in range(100):
        f_new = churchill_friction_factor(Re_test, roughness / D_inner)
        if abs(f_new - f_old) < 1e-7: 
            return f_new
        f_old = f_new
        Re_test *= 10 # 극한으로 보냄
    return f_new

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, sigma_L, D_inner, angle_deg, roughness, P_Pa):
    """
    [Beggs and Brill (1973) Rigorous Model]
    Transition 영역의 가중 보간법(Interpolation) 적용 및
    액체 속도수(N_VL) 기반의 엄밀한 경사 보정 계수(Inclination Correction) 복원.
    하향(Downhill) 유동에 대한 완벽한 처리 및 CoolProp 표면장력(sigma_L) 연동.
    """
    v_m = v_SL + v_SG # 혼합물 유속
    if v_m <= 0: v_m = 1e-6
    lambda_L = max(v_SL / v_m, 1e-5) # 입력 액체 체적비 (No-slip holdup)
    
    # 무차원 수 계산 (관성력, 중력, 표면장력)
    N_Fr = max((v_m**2) / (9.81 * D_inner), 1e-5) # Froude Number
    N_VL = v_SL * ((rho_L / (9.81 * sigma_L))**0.25) # Liquid Velocity Number
    
    # 유동 양식 경계값 계산
    L1 = 316 * lambda_L**0.302
    L2 = 0.000925 * lambda_L**-2.4684
    L3 = 0.1 * lambda_L**-1.4516
    L4 = 0.5 * lambda_L**-6.738
    
    # 맵(Map)을 통한 유동 양식 판별
    regime = "Transition"
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2): regime = "Segregated"
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4): regime = "Intermittent"
    elif (lambda_L < 0.4 and N_Fr >= L1) or (lambda_L >= 0.4 and N_Fr > L4): regime = "Distributed"
    
    # 내부 헬퍼 함수: 특정 유동 양식에 대한 수평 홀드업 및 경사 보정계수 도출
    def get_holdup_and_C(reg_type):
        # 1. 수평 홀드업 계산
        if reg_type == "Segregated": a, b, c = 0.98, 0.4846, 0.0868
        elif reg_type == "Intermittent": a, b, c = 0.845, 0.5351, 0.0173
        else: a, b, c = 1.065, 0.5824, 0.0609 # Distributed
            
        H_L_0 = (a * lambda_L**b) / (N_Fr**c)
        H_L_0 = min(max(H_L_0, lambda_L), 1.0) # 물리적 한계치 방어
        
        # 2. 경사 보정계수 (C) 상수 결정 (Uphill vs Downhill 완벽 분리)
        if angle_deg >= 0: # Uphill (상향 유동 및 수평)
            if reg_type == "Segregated": d, e, f, g = 0.011, -3.768, 3.539, -1.614
            elif reg_type == "Intermittent": d, e, f, g = 2.96, 0.305, -0.4473, 0.0978
            else: d, e, f, g = 0, 0, 0, 0 # Distributed
        else: # Downhill (하향 유동 - 모든 Regime 공통 상수)
            d, e, f, g = 4.70, -0.3692, 0.1244, -0.5056
            
        # 경사 보정계수 수식 계산
        if d == 0: 
            C_val = 0
        else: 
            C_val = (1 - lambda_L) * math.log(max(d * (lambda_L**e) * (N_VL**f) * (N_Fr**g), 1e-5))
        
        return H_L_0, C_val

    # 유동 양식에 따른 실제 액체 홀드업(H_L) 결정 로직
    if regime == "Transition":
        # Transition 영역 보간법 (Segregated와 Intermittent 사이)
        H_L_0_seg, C_seg = get_holdup_and_C("Segregated")
        H_L_0_int, C_int = get_holdup_and_C("Intermittent")
        
        H_L_seg = H_L_0_seg * (1 + C_seg * math.sin(math.radians(angle_deg)))
        H_L_int = H_L_0_int * (1 + C_int * math.sin(math.radians(angle_deg)))
        
        A_weight = (L3 - N_Fr) / (L3 - L2) if L3 != L2 else 0.5
        H_L = A_weight * H_L_seg + (1 - A_weight) * H_L_int
    else:
        H_L_0, C_val = get_holdup_and_C(regime)
        H_L = H_L_0 * (1 + C_val * math.sin(math.radians(angle_deg)))
        
    H_L = min(max(H_L, 0.0), 1.0)

    # 2상 유동의 혼합 물성치 (밀도 및 점도)
    rho_n = rho_L * lambda_L + rho_G * (1 - lambda_L)
    rho_s = rho_L * H_L + rho_G * (1 - H_L)
    mu_n = mu_L * lambda_L + mu_G * (1 - lambda_L)
    Re_n = (rho_n * v_m * D_inner) / mu_n
    
    # 마찰 계수 및 2상 승수 보정 (e^S)
    f_n = churchill_friction_factor(Re_n, roughness/D_inner)
    y = lambda_L / (H_L**2) if H_L > 0 else 1.0
    y = max(y, 1e-5) # 로그 에러 방지
    
    if 1 < y < 1.2: S = math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4)
    else: S = math.log(y)
    f_tp = f_n * math.exp(S)
    
    # 중력 강하량(Elevation)과 가속도 강하 인자(Acceleration, E_k)
    dP_dl_elev = rho_s * 9.81 * math.sin(math.radians(angle_deg))
    E_k = (v_m * v_SG * rho_n) / P_Pa if P_Pa > 0 else 0
    
    return {"f_tp": f_tp, "rho_n": rho_n, "v_m": v_m, "dP_dl_elev": dP_dl_elev, "E_k": E_k, "flow_regime": regime, "lambda_L": lambda_L, "Re_n": Re_n}

def get_k_pipe_extrapolated(T_C, T_arr, k_arr):
    """
    [선형 외삽법(Linear Extrapolation) 기반 열전도도 도출]
    DB에 극저온 조건이 없을 경우 기울기를 바탕으로 값을 연장(외삽)합니다.
    """
    if T_C >= T_arr[0] and T_C <= T_arr[-1]: return np.interp(T_C, T_arr, k_arr)
    elif T_C < T_arr[0]: return k_arr[0] + ((k_arr[1] - k_arr[0]) / (T_arr[1] - T_arr[0])) * (T_C - T_arr[0])
    else: return k_arr[-1] + ((k_arr[-1] - k_arr[-2]) / (T_arr[-1] - T_arr[-2])) * (T_C - T_arr[-1])

def get_robust_prop(prop, T, P, fluid_string, fractions, default_val, tracker):
    """
    [혼합물(Mixture) 물성치 계산 안전장치 및 추적기(Audit Trail)]
    CoolProp이 지원하지 않는 혼합물 전달물성치(점도, 표면장력 등)에 대한 우회 기법입니다.
    """
    try:
        return PropsSI(prop, 'T', T, 'P', P, fluid_string)
    except Exception:
        fluids = [f.split('[')[0] for f in fluid_string.split('&')]
        if len(fluids) == len(fractions):
            val_mix = 0.0
            try:
                for i, f in enumerate(fluids):
                    val_mix += fractions[i] * PropsSI(prop, 'T', T, 'P', P, f)
                tracker.add(f"'{prop}' 혼합물 물성 에러 ➔ 단일성분 가중평균(Mixing Rule) 적용")
                return val_mix
            except: pass
        tracker.add(f"'{prop}' 물성 모델 완전 부재 ➔ 기본 상수({default_val}) 방어 로직 적용")
        return default_val

def calculate_heat_transfer(Re, Pr, k_fluid, D_in, D_out, k_pipe, k_ins, t_ins, h_ext):
    """
    [1차원 반경 방향 열 저항 네트워크(Thermal Resistance Network)]
    열전달 저항들을 직렬 합산하여 총괄 열전달 계수(U)를 구합니다.
    """
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36 # 대류 열전달
    h_in = (Nu * k_fluid) / D_in if D_in > 0 else 1000
    R_conv_in = 1.0 / (h_in * math.pi * D_in)
    R_cond_pipe = math.log(D_out / D_in) / (2 * math.pi * k_pipe) if D_out > D_in else 0
    D_ins_out = D_out + 2 * t_ins
    R_cond_ins = math.log(D_ins_out / D_out) / (2 * math.pi * k_ins) if t_ins > 0 else 0
    R_conv_out = 1.0 / (h_ext * math.pi * D_ins_out)
    
    R_total = R_conv_in + R_cond_pipe + R_cond_ins + R_conv_out
    return 1.0 / (R_total * math.pi * D_ins_out) if R_total > 0 else 0

# ==========================================
# 2. 핵심 물리 엔진 (HYSYS 3중 중첩 루프)
# ==========================================

def solve_inner_loop_pressure(T_in, P_in, T_out_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, audit_tracker):
    """
    [Inner Loop: 운동량 수지를 통한 압력 수렴 알고리즘 (할선법 적용)]
    Middle Loop에서 추정한 '출구 온도'에서, 마찰력과 고저차를 만족하는 진짜 출구 압력을 역산.
    """
    P0 = P_in              
    P1 = P_in - 500        
    tol = 100              
    
    def calc_P_out(P_guess):
        P_avg = (P_in + P_guess) / 2.0
        T_avg = (T_in + T_out_guess) / 2.0
        
        if P_avg < 10000: raise ValueError(f"내부 압력이 너무 낮습니다 ({P_avg/1e5:.3f} bar). 배관경을 키우거나 유량을 줄이세요.")
            
        try: phase_raw = PhaseSI('T', T_avg, 'P', P_avg, fluid_string)
        except: phase_raw = "unknown"
        
        Q_val = -1.0; is_twophase = False
        if phase_raw == 'twophase':
            is_twophase = True
            try: Q_val = PropsSI('Q', 'T', T_avg, 'P', P_avg, fluid_string)
            except: is_twophase = False
            
        if not is_twophase: # 단상 유동
            rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
            mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
            vel = mass_flow / (rho * A_cross)
            Re = (rho * vel * D_inner) / mu
            f_factor = churchill_friction_factor(Re, roughness/D_inner)
            
            dP_total = ((f_factor * rho * vel**2) / (2 * D_inner)) * dL + (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
            regime = "1-Phase Liquid" if rho > 300 else "1-Phase Gas"
        else: # 2상 유동 (Beggs & Brill)
            rho_L = PropsSI('D', 'P', P_avg, 'Q', 0, fluid_string)
            rho_G = PropsSI('D', 'P', P_avg, 'Q', 1, fluid_string)
            mu_L = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-3, audit_tracker)
            mu_G = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
            
            # ⭐️ CoolProp에서 실제 표면장력 호출 시도, 에러 시 0.02 방어 ⭐️
            sigma_L = get_robust_prop('I', T_avg, P_avg, fluid_string, norm_fractions, 0.02, audit_tracker)
            
            v_SG = (mass_flow * Q_val) / (rho_G * A_cross)
            v_SL = (mass_flow * (1 - Q_val)) / (rho_L * A_cross)
            
            bb = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, sigma_L, D_inner, angle_deg, roughness, P_avg)
            dP_fric = ((bb["f_tp"] * bb["rho_n"] * bb["v_m"]**2) / (2 * D_inner)) * dL
            dP_total = (dP_fric + bb["dP_dl_elev"] * dL) / (1 - bb["E_k"])
            regime = bb["flow_regime"]
            
        return P_in - dP_total, is_twophase, Q_val, regime

    P_calc0, is_tp0, Q0, reg0 = calc_P_out(P0)
    f0 = P_calc0 - P0 
    if abs(f0) < tol: return P_calc0, is_tp0, Q0, reg0
        
    P_calc1, is_tp1, Q1, reg1 = calc_P_out(P1)
    f1 = P_calc1 - P1
    
    for _ in range(20): # 할선법
        if abs(f1) < tol: return P_calc1, is_tp1, Q1, reg1
        
        if abs(f1 - f0) < 1e-5: P_new = P1 - f1 * 0.5 
        else: P_new = P1 - f1 * ((P1 - P0) / (f1 - f0))
        
        P0, f0 = P1, f1; P1 = P_new
        P_calc1, is_tp1, Q1, reg1 = calc_P_out(P1)
        f1 = P_calc1 - P1

    return P1, is_tp1, Q1, reg1 

def solve_middle_loop_temp(T_in, P_in, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, D_outer, roughness, angle_deg, dL, k_pipe, k_ins, t_ins, h_ext, T_amb_K, audit_tracker):
    """
    [Middle Loop: 에너지 수지를 통한 온도 수렴 알고리즘 (PH-Flash 기반)]
    가상의 출구 온도에서 발생하는 열교환량(Q)을 엔탈피에 반영해, 상태방정식이 말하는 진짜 온도를 할선법으로 찾음.
    """
    T0 = T_in          
    T1 = T_in - 0.1    
    tol = 0.01         
    
    try: 
        H_in = PropsSI('H', 'T', T_in, 'P', P_in, fluid_string)
        use_enthalpy = True 
    except: 
        use_enthalpy = False
        
    def calc_T_out(T_guess):
        P_out_calc, is_tp, Q_val, regime = solve_inner_loop_pressure(T_in, P_in, T_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, audit_tracker)
        P_avg = (P_in + P_out_calc) / 2.0; T_avg = (T_in + T_guess) / 2.0
        
        Cp = get_robust_prop('C', T_avg, P_avg, fluid_string, norm_fractions, 2000, audit_tracker)
        k_fluid = get_robust_prop('L', T_avg, P_avg, fluid_string, norm_fractions, 0.1, audit_tracker)
        mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
        try: rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
        except: rho = 500
            
        Re = (rho * (mass_flow / (rho * A_cross)) * D_inner) / mu
        Pr = (Cp * mu) / k_fluid if k_fluid > 0 else 1.0
        
        U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe, k_ins, t_ins, h_ext)
        Q_heat = U * math.pi * D_outer * dL * (T_amb_K - T_avg) 
        
        if use_enthalpy:
            try: 
                return PropsSI('T', 'H', H_in + Q_heat / mass_flow, 'P', P_out_calc, fluid_string), P_out_calc, is_tp, Q_val, regime
            except: pass
        return T_in + Q_heat / (mass_flow * Cp), P_out_calc, is_tp, Q_val, regime

    T_calc0, P_calc0, is_tp0, Q0, reg0 = calc_T_out(T0)
    f0 = T_calc0 - T0
    if abs(f0) < tol: return T_calc0, P_calc0, is_tp0, Q0, reg0
        
    T_calc1, P_calc1, is_tp1, Q1, reg1 = calc_T_out(T1)
    f1 = T_calc1 - T1
    
    for _ in range(20):
        if abs(f1) < tol: return T_calc1, P_calc1, is_tp1, Q1, reg1
        
        if abs(f1 - f0) < 1e-6: T_new = T1 - f1 * 0.5
        else: T_new = T1 - f1 * ((T1 - T0) / (f1 - f0))
        
        T0, f0 = T1, f1; T1 = T_new
        T_calc1, P_calc1, is_tp1, Q1, reg1 = calc_T_out(T1)
        f1 = T_calc1 - T1

    return T1, P_calc1, is_tp1, Q1, reg1

# ==========================================
# 3. UI 렌더링 및 동적 컴포넌트 추가
# ==========================================

st.title("⚓ 해양/플랜트 다상 유동 파이프라인 시뮬레이터 (V6.0)")
st.markdown("**(HYSYS 3-Nested-Loop, 디커플링 피팅 물리엔진, PH-Flash 열수지 적용)**")

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

st.header("1. 유체 성분 및 운전 조건")
col1, col2, col3 = st.columns(3)
selected_fluids = col1.multiselect("유체 성분 선택 (CoolProp 기반)", ["Methane", "Ethane", "Propane", "Ammonia", "Water", "Nitrogen", "CO2"], default=["Methane"])

fractions = []
if selected_fluids:
    st.write("몰 분율(Mole Fraction) 입력 (자동 정규화됨):")
    cols_frac = st.columns(len(selected_fluids))
    for i, fluid in enumerate(selected_fluids):
        frac = cols_frac[i].number_input(f"{fluid}", min_value=0.0, value=1.0, key=f"frac_{fluid}")
        fractions.append(frac)

T_inlet_C = col2.number_input("입구 온도 (°C)", value=20.0)
P_inlet_bar = col3.number_input("입구 압력 (bar)", value=10.0)
mass_flow = st.number_input("질량 유량 (kg/s)", value=5.0)

st.header("2. 순차적 파이프라인 빌더")
comp_type = st.radio("추가할 컴포넌트", ["Pipe Segment (배관)", "Fitting / Valve (밸브 및 피팅)"], horizontal=True)

if "Pipe" in comp_type:
    ac1, ac2, ac3 = st.columns(3)
    p_len = ac1.number_input("직관 길이 (m)", min_value=0.0, value=10.0, format="%.4f")
    p_elev_type = ac2.selectbox("경사 입력 방식", ["Angle (deg)", "Height (m)"])
    p_elev_val = ac3.number_input("경사/높이 값", value=0.0, format="%.4f")
    
    pc1, pc2, pc3 = st.columns(3)
    D_inner = pc1.number_input("내부 직경 (m)", value=0.1, format="%.4f")
    thickness = pc2.number_input("배관 두께 (m)", value=0.005, format="%.4f")
    selected_sys_material = pc3.selectbox("배관 재질 선택 (ASME)", list(material_db.MATERIAL_MAP.keys()))
    
    if st.button("➕ 배관 추가", type="secondary"):
        st.session_state.pipeline.append({
            "type": "Pipe", "length": p_len, "elev_type": p_elev_type, "elev_val": p_elev_val,
            "D_inner": D_inner, "thickness": thickness, "material": selected_sys_material
        })
        st.rerun()
else:
    fc1, fc2 = st.columns(2)
    f_type = fc1.selectbox("피팅/밸브 종류", list(FITTING_DB.keys()))
    f_qty = fc2.number_input("수량", min_value=1, value=1, step=1)
    
    if st.button("➕ 피팅/밸브 추가", type="secondary"):
        st.session_state.pipeline.append({
            "type": "Fitting", "name": f_type, "qty": f_qty
        })
        st.rerun()

if st.session_state.pipeline:
    st.markdown("##### 🧱 현재 구성된 파이프라인 목록 (Flow: 위 ➔ 아래)")
    h1, h2, h3 = st.columns([0.1, 0.7, 0.2])
    h1.caption("순서")
    h2.caption("컴포넌트 상세 제원")
    h3.caption("이동 / 삭제")
    st.divider()
    
    for idx, comp in enumerate(st.session_state.pipeline):
        c1, c2, c3, c4, c5 = st.columns([0.1, 0.7, 0.06, 0.06, 0.08])
        c1.write(f"**[{idx+1}]**")
        
        if comp.get("type") == "Pipe":
            _len = comp.get('length', 0)
            _id = comp.get('D_inner', 0)
            _mat = comp.get('material', '').split(' / ')[0] if 'material' in comp else "Unknown"
            c2.write(f"**배관:** L={_len}m | ID={_id}m | 재질: {_mat}")
        else:
            _name = comp.get('name', 'Unknown')
            _qty = comp.get('qty', 1)
            c2.write(f"**피팅/밸브:** {_name} (x{_qty})")
            
        if c3.button("⬆️", key=f"up_{idx}") and idx > 0:
            st.session_state.pipeline[idx-1], st.session_state.pipeline[idx] = st.session_state.pipeline[idx], st.session_state.pipeline[idx-1]
            st.rerun()
        if c4.button("⬇️", key=f"down_{idx}") and idx < len(st.session_state.pipeline)-1:
            st.session_state.pipeline[idx+1], st.session_state.pipeline[idx] = st.session_state.pipeline[idx], st.session_state.pipeline[idx+1]
            st.rerun()
        if c5.button("❌", key=f"del_{idx}"):
            st.session_state.pipeline.pop(idx)
            st.rerun()
    st.divider()

st.header("3. 외부 환경 (전체 공통 적용)")
col_env1, col_env2, col_env3 = st.columns(3)
T_amb_C = col_env1.number_input("대기 온도 (°C)", value=25.0)
h_ext = col_env2.number_input("외부 대류 열전달 계수 (W/m²K)", value=10.0)
N_per_pipe = col_env3.number_input("구간(Pipe)당 분할 노드 수", min_value=1, value=10, step=1, help="Middle Loop 수렴으로 5~10 분할만으로도 HYSYS급 초정밀 해를 보장합니다.")

col_env4, col_env5 = st.columns(2)
t_ins = col_env4.number_input("보온재 두께 (m)", min_value=0.0, value=0.0, format="%.4f")
k_ins = col_env5.number_input("보온재 열전도도 (W/m·K)", value=0.035, format="%.4f")

# ==========================================
# 4. 시뮬레이션 메인 루프 (Outer Loop)
# ==========================================

if st.button("🚀 해석 실행 (Run Simulator)", type="primary"):
    if not selected_fluids or not st.session_state.pipeline:
        st.error("유체 성분과 파이프라인 컴포넌트를 최소 1개 이상 추가해야 합니다.")
        st.stop()
        
    total_frac = sum(fractions)
    norm_fractions = [f / total_frac for f in fractions]
    fluid_string = "&".join([f"{f}[{frac}]" for f, frac in zip(selected_fluids, norm_fractions)])
    if len(selected_fluids) == 1: fluid_string = selected_fluids[0]

    global_audit_tracker = set()
    results = []
    
    T_current_K = T_inlet_C + 273.15
    P_current_Pa = P_inlet_bar * 100000
    L_cum = 0.0
    Z_cum = 0.0 
    
    results.append({
        "Component": "Inlet", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, 
        "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, 
        "Phase": "-", "dP (bar)": 0.0, "Regime": "Inlet"
    })
    
    curr_D_inner, curr_thickness, curr_roughness, curr_mat_info = 0.1, 0.005, 4.5e-5, list(material_db.MATERIAL_MAP.values())[0]
    status_box = st.status("🤖 V6 디커플링 물리 엔진 및 PH-Flash 해석 진행 중...", expanded=True)
    
    try:
        for comp_idx, comp in enumerate(st.session_state.pipeline):
            
            # ----------------------------------------
            # A. 직관(Pipe) 해석 블록: 3중 중첩 루프 적용
            # ----------------------------------------
            if comp.get("type") == "Pipe":
                status_box.update(label=f"🔄 [Pipe {comp_idx+1}/{len(st.session_state.pipeline)}] 3중 루프(Outer-Middle-Inner) P-T 수렴 계산 중...")
                
                curr_D_inner = comp.get("D_inner", curr_D_inner)
                curr_thickness = comp.get("thickness", curr_thickness)
                if "material" in comp: curr_mat_info = material_db.MATERIAL_MAP[comp["material"]]
                curr_roughness = curr_mat_info["roughness_m"]
                asme_table = material_db.RAW_DB[curr_mat_info["asme_category"]][curr_mat_info["asme_grade"]]
                D_outer = curr_D_inner + 2 * curr_thickness
                A_cross = math.pi * (curr_D_inner / 2)**2
                
                length = comp.get("length", 10.0)
                dL = length / N_per_pipe if N_per_pipe > 0 else 0
                
                if comp.get("elev_type", "Angle (deg)") == "Height (m)":
                    angle_deg = 0 if length == 0 else math.degrees(math.asin(max(min(comp.get("elev_val", 0.0) / length, 1.0), -1.0)))
                else:
                    angle_deg = comp.get("elev_val", 0.0)
                dZ = dL * math.sin(math.radians(angle_deg))

                # 파이프를 노드 단위(dL)로 쪼개어 연속적으로 마칭(Marching)
                for i in range(N_per_pipe):
                    k_pipe_current = get_k_pipe_extrapolated(T_current_K - 273.15, asme_table["T_C"], asme_table["k_W_mK"])
                    
                    T_out, P_out, is_tp, Q_val, flow_regime = solve_middle_loop_temp(
                        T_current_K, P_current_Pa, fluid_string, norm_fractions, mass_flow, A_cross, 
                        curr_D_inner, D_outer, curr_roughness, angle_deg, dL, 
                        k_pipe_current, k_ins, t_ins, h_ext, T_amb_C + 273.15, global_audit_tracker
                    )
                    
                    dP_step = P_current_Pa - P_out
                    P_current_Pa = P_out
                    T_current_K = T_out
                    L_cum += dL
                    Z_cum += dZ
                    results.append({
                        "Component": f"Pipe_{comp_idx+1}", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, 
                        "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, 
                        "Phase": "2-Phase" if is_tp else "1-Phase", "dP (bar)": dP_step / 1e5, 
                        "Regime": flow_regime
                    })

            # ----------------------------------------
            # B. 피팅/밸브(Fitting) 해석 블록: 국부 저항 독립 계산
            # ----------------------------------------
            elif comp.get("type") == "Fitting":
                f_name = comp.get("name", "Unknown")
                qty = comp.get("qty", 1)
                status_box.update(label=f"🔄 [Fitting {comp_idx+1}/{len(st.session_state.pipeline)}] {f_name} 국부 저항 독립 계산 중...")
                
                if P_current_Pa < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 통과 전 압력이 {P_current_Pa/1e5:.3f} bar로 너무 낮습니다.")

                A_cross = math.pi * (curr_D_inner / 2)**2
                fit_data = FITTING_DB.get(f_name, {"A": 0.0, "B": 30, "Chisholm_B": 1.5})
                
                # Crane TP-410 방법론: f_T 도출 및 K-Factor 산출
                f_T = calculate_fT_hysys(curr_roughness, curr_D_inner)
                K_factor = (fit_data["A"] + fit_data["B"] * f_T) * qty
                
                try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                except: phase_raw = "unknown"
                is_twophase = (phase_raw == 'twophase')
                
                if not is_twophase:
                    # 단상 밸브 압력 강하 (단순 동압 비례 모델)
                    rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    vel = mass_flow / (rho * A_cross)
                    dP_fit = K_factor * rho * (vel**2) / 2.0
                    regime_label = "1-Phase Liquid" if rho > 300 else "1-Phase Gas"
                else:
                    # 2상 밸브 압력 강하 (Chisholm B-parameter 모델)
                    Q_val = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    G_mass_flux = mass_flow / A_cross
                    
                    dP_LO = K_factor * (G_mass_flux**2) / (2 * rho_L) 
                    phi_LO2 = 1 + (rho_L / rho_G - 1) * (fit_data["Chisholm_B"] * Q_val * (1 - Q_val) + Q_val**2)
                    dP_fit = dP_LO * phi_LO2
                    regime_label = "2-Phase (Chisholm)"
                
                P_out_fit = P_current_Pa - dP_fit
                if P_out_fit < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 밸브 통과 후 진공({P_out_fit/1e5:.3f} bar)에 도달!")
                
                # PH-Flash 기반 등엔탈피 팽창 (Isenthalpic Throttling / Joule-Thomson 냉각)
                try:
                    H_in = PropsSI('H', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    T_out_fit = PropsSI('T', 'H', H_in, 'P', P_out_fit, fluid_string)
                except:
                    T_out_fit = T_current_K
                
                P_current_Pa = P_out_fit
                T_current_K = T_out_fit
                
                results.append({
                    "Component": f"Fitting_{comp_idx+1} ({f_name})", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, 
                    "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, 
                    "Phase": "2-Phase" if is_twophase else "1-Phase", "dP (bar)": dP_fit / 1e5, 
                    "Regime": regime_label
                })

        status_box.update(label="✅ 최고 수준의 물리 모델 해석(V6.0) 완료!", state="complete")
        
    except ValueError as ve:
        status_box.update(label="🚨 물리적 한계 도달 (시뮬레이션 중단)", state="error")
        st.error(f"{ve}")
        st.stop()
    except Exception as e:
        status_box.update(label="🚨 시스템 오류 발생", state="error")
        st.error(f"예상치 못한 에러: {e}")
        st.stop()
        
    if global_audit_tracker:
        st.info("⚠️ **[물성치 Fallback 알림]** 특정 구간에서 물리적 한계로 인해 다음 가정이 사용되었습니다.\n\n" + "\n".join([f"- {msg}" for msg in global_audit_tracker]))

    df_res = pd.DataFrame(results)
    st.subheader("📊 시뮬레이션 결과 데이터")
    st.dataframe(df_res.style.format({"P (bar)": "{:.3f}", "T (°C)": "{:.2f}", "dP (bar)": "{:.4f}", "L_cum (m)": "{:.2f}", "Z_cum (m)": "{:.2f}"}), use_container_width=True)

    st.subheader("📈 파이프라인 열/수력학적 프로파일")
    
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["Z_cum (m)"], mode='lines', line=dict(color='gray', width=4), name='Pipeline'))
    
    fittings_df = df_res[df_res['Component'].str.contains("Fitting")]
    if not fittings_df.empty:
        fig_2d.add_trace(go.Scatter(
            x=fittings_df["L_cum (m)"], y=fittings_df["Z_cum (m)"],
            mode='markers', marker=dict(color='red', size=12, symbol='diamond'),
            name='Valve/Fitting', hovertext=fittings_df['Component']
        ))
    
    fig_2d.update_layout(title_text="파이프라인 2D 스케치 (Side View)", xaxis_title_text="누적 길이 (m)", yaxis_title_text="고도 (m)", showlegend=True)
    
    fig_pt = go.Figure()
    fig_pt.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["P (bar)"], mode='lines', name='Pressure (bar)', yaxis='y1', line=dict(color='blue', width=3)))
    fig_pt.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["T (°C)"], mode='lines', name='Temperature (°C)', yaxis='y2', line=dict(color='red', width=3, dash='dash')))
    
    fig_pt.update_layout(
        title=dict(text="압력 및 온도 프로파일 (Valve 통과 시 급격한 하강 주목)"),
        xaxis=dict(title=dict(text="누적 길이 (m)")),
        yaxis=dict(
            title=dict(text="압력 (bar)", font=dict(color="blue")),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title=dict(text="온도 (°C)", font=dict(color="red")),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right"
        ),
        hovermode="x unified"
    )
    
    tc1, tc2 = st.columns(2)
    tc1.plotly_chart(fig_2d, use_container_width=True)
    tc2.plotly_chart(fig_pt, use_container_width=True)
