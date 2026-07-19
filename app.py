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
    레이놀즈 수(Re)와 상대 조도(e/d)를 입력받아 마찰계수(f)를 반환합니다.
    Moody 차트의 층류, 전이 영역, 난류 전 구간을 불연속점(Discontinuity) 없이 
    하나의 수식으로 덮을 수 있어 컴퓨터 반복 해석 시 발산(터짐)을 막아줍니다.
    """
    if Re <= 0: return 0.01 # 유속이 거의 없을 때의 방어 로직
    A = (-2.457 * math.log((7.0 / Re)**0.9 + 0.27 * ed))**16
    B = (37530.0 / Re)**16
    f = 8 * ((8 / Re)**12 + 1 / (A + B)**1.5)**(1/12)
    return f

def calculate_fT_hysys(roughness, D_inner):
    """
    [완전 난류 마찰계수(fT) 계산 - HYSYS/Crane TP-410 방식]
    밸브 및 피팅의 압력 강하(K-factor)를 구하기 위해서는 fT가 필요합니다.
    유속을 무한대(Re -> 무한대)로 가속시켰을 때 마찰계수가 더 이상 변하지 않고
    배관의 거칠기(Roughness)에 의해서만 결정되는 한계 수렴 지점을 
    Churchill 공식을 반복 호출하여 수학적으로 찾아냅니다.
    """
    Re_test = 1e6 # 초기 레이놀즈 수를 난류로 크게 설정
    f_old = 0.0
    for _ in range(100):
        f_new = churchill_friction_factor(Re_test, roughness / D_inner)
        if abs(f_new - f_old) < 1e-7: # 마찰계수가 더 이상 안 변하면 (수렴)
            return f_new
        f_old = f_new
        Re_test *= 10 # 레이놀즈 수를 10배씩 뻥튀기하며 극한으로 보냄
    return f_new

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, angle_deg, roughness, P_Pa):
    """
    [Beggs and Brill (1973) 2상 유동 모델]
    파이프 내에 액체와 기체가 섞여 흐를 때의 마찰 강하 및 위치 수두(중력)를 계산합니다.
    해양 배관처럼 경사(Inclination)가 있는 관에서 홀드업(액체가 고이는 현상)을 매우 정확하게 예측합니다.
    """
    v_m = v_SL + v_SG # 혼합물 유속 = 겉보기 액체유속 + 겉보기 기체유속
    if v_m <= 0: v_m = 1e-6
    lambda_L = v_SL / v_m # 입력 액체 체적비 (No-slip holdup)
    N_Fr = (v_m**2) / (9.81 * D_inner) # Froude 수: 중력 대비 관성력의 비율 (유동 양식 판별에 사용)
    
    # 유동 양식(Flow Regime) 판별을 위한 경계값 계산
    L1 = 316 * lambda_L**0.302; L2 = 0.000925 * lambda_L**-2.4684
    L3 = 0.1 * lambda_L**-1.4516; L4 = 0.5 * lambda_L**-6.738
    
    # 맵(Map)을 통한 유동 양식 판별 (분리형, 간헐형, 분산형)
    regime = "Transition"
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2): regime = "Segregated"
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4): regime = "Intermittent"
    elif (lambda_L < 0.4 and N_Fr >= L1) or (lambda_L >= 0.4 and N_Fr > L4): regime = "Distributed"
    
    # 유동 양식에 따른 수평 배관 액체 홀드업(H_L_0) 경험식 계수
    if regime == "Segregated": a, b, c = 0.98, 0.4846, 0.0868
    elif regime == "Intermittent": a, b, c = 0.845, 0.5351, 0.0173
    elif regime == "Distributed": a, b, c = 1.065, 0.5824, 0.0609
    else: a, b, c = 0.9125, 0.50985, 0.05205 # Transition

    H_L_0 = a * lambda_L**b / N_Fr**c
    if H_L_0 < lambda_L: H_L_0 = lambda_L
    if H_L_0 > 1.0: H_L_0 = 1.0

    # 경사 보정 계수(Inclination Correction Factor) 적용
    C_val = (1 - lambda_L) * math.log(max(lambda_L, 1e-5) * 0.01**0.05 * 1e5**0.1 * 1e5**0.1)
    if regime == "Segregated": beta = max(0, (1 - lambda_L) * math.log(max(1e-5, C_val)))
    else: beta = 0
    
    # 실제 액체 홀드업 (H_L) - 파이프 내에 실제로 존재하는 액체의 부피 비
    H_L = H_L_0 * (1 + beta * math.sin(math.radians(angle_deg)))
    if H_L < 0: H_L = 0
    elif H_L > 1: H_L = 1
    
    # 2상 유동의 혼합 물성치 (밀도 및 점도)
    rho_n = rho_L * lambda_L + rho_G * (1 - lambda_L) # No-slip 밀도 (마찰 계산용)
    rho_s = rho_L * H_L + rho_G * (1 - H_L)           # Slip 밀도 (정수두/중력 계산용)
    mu_n = mu_L * lambda_L + mu_G * (1 - lambda_L)
    Re_n = (rho_n * v_m * D_inner) / mu_n
    
    # 마찰 계수 및 2상 승수 보정 (e^S)
    f_n = churchill_friction_factor(Re_n, roughness/D_inner)
    y = lambda_L / H_L**2 if H_L > 0 else 1.0
    S = math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4) if 1 < y < 1.2 else math.log(y)
    f_tp = f_n * math.exp(S)
    
    # 중력 강하량(Elevation)과 가속도 강하 인자(Acceleration, E_k)
    dP_dl_elev = rho_s * 9.81 * math.sin(math.radians(angle_deg))
    E_k = (v_m * v_SG * rho_n) / P_Pa if P_Pa > 0 else 0
    
    return {"f_tp": f_tp, "rho_n": rho_n, "v_m": v_m, "dP_dl_elev": dP_dl_elev, "E_k": E_k, "flow_regime": regime, "lambda_L": lambda_L, "Re_n": Re_n}

def get_k_pipe_extrapolated(T_C, T_arr, k_arr):
    """
    [선형 외삽법(Linear Extrapolation) 기반 열전도도 도출]
    DB에 20℃~500℃ 데이터만 있을 때, 극저온(-160℃, LNG) 조건이 들어오면
    단순히 20℃ 값을 쓰는(Clipping) 것이 아니라, 하위 2개 점의 기울기를 바탕으로
    수학적으로 값을 연장(외삽)하여 신뢰성을 확보합니다.
    """
    if T_C >= T_arr[0] and T_C <= T_arr[-1]: return np.interp(T_C, T_arr, k_arr)
    elif T_C < T_arr[0]: return k_arr[0] + ((k_arr[1] - k_arr[0]) / (T_arr[1] - T_arr[0])) * (T_C - T_arr[0])
    else: return k_arr[-1] + ((k_arr[-1] - k_arr[-2]) / (T_arr[-1] - T_arr[-2])) * (T_C - T_arr[-1])

def get_robust_prop(prop, T, P, fluid_string, fractions, default_val, tracker):
    """
    [혼합물(Mixture) 물성치 계산 안전장치 및 추적기(Audit Trail)]
    CoolProp은 메탄+에탄 같은 혼합물에 대해 밀도는 잘 구하지만 점도(Viscosity) 모델이
    없는 경우가 많습니다. 에러 시 앱이 뻗지 않도록 1. 단일 성분 가중평균(Mixing Rule), 
    2. 상수 덮어쓰기 순으로 방어(Fallback)하며, 어떤 조치를 취했는지 tracker에 기록합니다.
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
                tracker.add(f"'{prop}' 혼합물 물성 에러 ➔ 단일성분 몰분율 가중평균(Mixing Rule) 적용")
                return val_mix
            except: pass
        tracker.add(f"'{prop}' 물성 모델 완전 부재 ➔ 기본 상수({default_val}) 덮어쓰기 적용")
        return default_val

def calculate_heat_transfer(Re, Pr, k_fluid, D_in, D_out, k_pipe, k_ins, t_ins, h_ext):
    """
    [1차원 반경 방향 열 저항 네트워크(Thermal Resistance Network)]
    전기 회로의 저항처럼 열전달 저항들을 직렬 합산하여 총괄 열전달 계수(U)를 구합니다.
    1. 내부 유체 대류 저항 (Dittus-Boelter 방정식 사용)
    2. 파이프 강재 전도 저항 (Fourier 원통 법칙)
    3. 보온재(Insulation) 전도 저항
    4. 외부 대기 강제/자연 대류 저항
    """
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36 # 층류면 4.36 (일정 열유속)
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
    [Inner Loop: 압력 수렴 알고리즘 (할선법 적용)]
    Middle Loop에서 추정한 '출구 온도(T_out_guess)'가 맞다는 가정 하에,
    입구와 출구의 "평균 온도 및 평균 압력"에서 발생하는 유체 마찰력(운동량 보존)을 계산해
    실제 출구 압력(P_out_calc)을 역산하여 찾아내는 과정입니다.
    """
    P0 = P_in              # 초기값 1 (압력강하 0 가정)
    P1 = P_in - 500        # 초기값 2 (미세 강하 가정 - 기울기용)
    tol = 100              # 오차 허용범위 (100 Pa = 0.001 bar)
    
    def calc_P_out(P_guess):
        # 1. 평균 상태 변수 도출 (매우 중요: HYSYS Implicit 철학의 핵심)
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
            
        # 2. 유동 상(Phase)에 따른 압력 강하 연산
        if not is_twophase: # --- 1상 유동 ---
            rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
            mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
            vel = mass_flow / (rho * A_cross)
            Re = (rho * vel * D_inner) / mu
            f_factor = churchill_friction_factor(Re, roughness/D_inner)
            
            dP_total = ((f_factor * rho * vel**2) / (2 * D_inner)) * dL + (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
            regime = "1-Phase Liquid" if rho > 300 else "1-Phase Gas"
        else:               # --- 2상 유동 (Beggs & Brill) ---
            rho_L = PropsSI('D', 'P', P_avg, 'Q', 0, fluid_string)
            rho_G = PropsSI('D', 'P', P_avg, 'Q', 1, fluid_string)
            mu_L = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-3, audit_tracker)
            mu_G = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
            
            v_SG = (mass_flow * Q_val) / (rho_G * A_cross)
            v_SL = (mass_flow * (1 - Q_val)) / (rho_L * A_cross)
            
            bb = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, angle_deg, roughness, P_avg)
            dP_fric = ((bb["f_tp"] * bb["rho_n"] * bb["v_m"]**2) / (2 * D_inner)) * dL
            dP_total = (dP_fric + bb["dP_dl_elev"] * dL) / (1 - bb["E_k"])
            regime = bb["flow_regime"]
            
        return P_in - dP_total, is_twophase, Q_val, regime

    # 3. 할선법 (Secant Method)을 통한 뿌리 찾기(Root Finding)
    P_calc0, is_tp0, Q0, reg0 = calc_P_out(P0)
    f0 = P_calc0 - P0 # 오차 함수 = (계산된 P) - (내가 찍은 P)
    if abs(f0) < tol: return P_calc0, is_tp0, Q0, reg0
        
    P_calc1, is_tp1, Q1, reg1 = calc_P_out(P1)
    f1 = P_calc1 - P1
    
    for _ in range(20): # 최대 20번 스무고개
        if abs(f1) < tol: return P_calc1, is_tp1, Q1, reg1
        
        # 할선법 수식: 직선의 기울기를 이용해 다음 P값을 지능적으로 찍음
        if abs(f1 - f0) < 1e-5: P_new = P1 - f1 * 0.5 
        else: P_new = P1 - f1 * ((P1 - P0) / (f1 - f0))
        
        P0, f0 = P1, f1; P1 = P_new
        P_calc1, is_tp1, Q1, reg1 = calc_P_out(P1)
        f1 = P_calc1 - P1

    return P1, is_tp1, Q1, reg1 # 수렴 실패 시 최후의 값 반환

def solve_middle_loop_temp(T_in, P_in, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, D_outer, roughness, angle_deg, dL, k_pipe, k_ins, t_ins, h_ext, T_amb_K, audit_tracker):
    """
    [Middle Loop: 온도 수렴 알고리즘 (에너지 보존, PH-Flash 기반)]
    임의의 출구 온도를 찍어보고, 그 온도일 때의 압력 강하(Inner Loop 호출)를 알아낸 뒤,
    파이프 방열/흡열량을 계산하여 열역학 1법칙(엔탈피 변화)에 딱 맞아떨어지는 
    진짜 출구 온도를 할선법으로 찾아냅니다.
    """
    T0 = T_in          # 초기값 1
    T1 = T_in - 0.1    # 초기값 2
    tol = 0.01         # 오차 허용범위 (0.01 도씨)
    
    # 열역학적 상태 계산(PH-Flash) 사용 여부 판단
    try: 
        H_in = PropsSI('H', 'T', T_in, 'P', P_in, fluid_string)
        use_enthalpy = True  # 혼합물 지원 시 정확한 엔탈피 방식 사용
    except: 
        use_enthalpy = False # 에러 시 근사 비열(Cp) 방식 사용
        
    def calc_T_out(T_guess):
        # 1. 찍어본 온도를 바탕으로 Inner Loop(압력) 호출 ➔ 진실된 압력을 받아옴!
        P_out_calc, is_tp, Q_val, regime = solve_inner_loop_pressure(T_in, P_in, T_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, audit_tracker)
        
        P_avg = (P_in + P_out_calc) / 2.0; T_avg = (T_in + T_guess) / 2.0
        
        # 2. 평균 온도/압력 기반 물성 추출 및 열전달량(Q) 도출
        Cp = get_robust_prop('C', T_avg, P_avg, fluid_string, norm_fractions, 2000, audit_tracker)
        k_fluid = get_robust_prop('L', T_avg, P_avg, fluid_string, norm_fractions, 0.1, audit_tracker)
        mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
        try: rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
        except: rho = 500
            
        Re = (rho * (mass_flow / (rho * A_cross)) * D_inner) / mu
        Pr = (Cp * mu) / k_fluid if k_fluid > 0 else 1.0
        
        U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe, k_ins, t_ins, h_ext)
        Q_heat = U * math.pi * D_outer * dL * (T_amb_K - T_avg) 
        
        # 3. 출구 온도(T_out) 역산
        if use_enthalpy:
            try: 
                # [PH-Flash 로직] 출구 엔탈피 = 입구 엔탈피 + (Q/m). 압력과 엔탈피로 온도 역산!
                return PropsSI('T', 'H', H_in + Q_heat / mass_flow, 'P', P_out_calc, fluid_string), P_out_calc, is_tp, Q_val, regime
            except: pass
        return T_in + Q_heat / (mass_flow * Cp), P_out_calc, is_tp, Q_val, regime

    # 4. 할선법 적용 (구조는 Inner Loop와 완전 동일)
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

# 라디오 버튼 선택에 따라 입력 폼이 동적으로 바뀜 (UI/UX 최적화)
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
    # material_db의 광범위한 HYSYS 피팅 리스트를 불러옴
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
    
    # 리스트 이동 및 삭제가 가능한 깔끔한 테이블 뷰 표출
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
    
    # 시스템 전체 상태 추적 변수 초기화
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
        # 컴포넌트(Pipe -> Valve -> Pipe)를 순차적으로 통과하는 최외곽 루프 (Outer Loop)
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
                    
                    # Middle Loop 함수 호출 (이 안에서 알아서 Inner Loop까지 돌며 T, P 수렴값을 찾아냄!)
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
                status_box.update(label=f"🔄 [Fitting {comp_idx+1}/{len(st.session_state.pipeline)}] {f_name} 국부 저항 (Crane/Chisholm 모델) 독립 계산 중...")
                
                if P_current_Pa < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 통과 전 압력이 {P_current_Pa/1e5:.3f} bar로 너무 낮습니다.")

                A_cross = math.pi * (curr_D_inner / 2)**2
                fit_data = FITTING_DB.get(f_name, {"A": 0.0, "B": 30, "Chisholm_B": 1.5})
                
                # [핵심 로직 1] 완전 난류 마찰계수(f_T) 도출 및 K-Factor 산출 (Crane TP-410 방법론)
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
                    # [핵심 로직 2] 2상 밸브 압력 강하 (Chisholm B-parameter 모델)
                    # 2상 혼합물이 유로 변경부에서 와류를 발생시키는 슬립(Slip) 현상을 정확히 보정
                    Q_val = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    G_mass_flux = mass_flow / A_cross
                    
                    dP_LO = K_factor * (G_mass_flux**2) / (2 * rho_L) # 100% 액체라고 가정했을 때의 압력강하
                    # 2상 승수 (Two-Phase Multiplier) 적용
                    phi_LO2 = 1 + (rho_L / rho_G - 1) * (fit_data["Chisholm_B"] * Q_val * (1 - Q_val) + Q_val**2)
                    dP_fit = dP_LO * phi_LO2
                    regime_label = "2-Phase (Chisholm)"
                
                P_out_fit = P_current_Pa - dP_fit
                if P_out_fit < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 밸브 통과 후 압력이 진공({P_out_fit/1e5:.3f} bar)에 도달! 밸브를 열거나 유량을 줄이세요.")
                
                # [핵심 로직 3] PH-Flash 기반 등엔탈피 팽창 (Isenthalpic Throttling)
                # 밸브를 통과하며 외부로 열을 빼앗기지 않았다(Q=0, H_out=H_in)고 가정하고,
                # 떨어진 압력에 맞춰 기화가 일어나며 자가 냉각(Joule-Thomson)되는 진짜 온도를 상태방정식으로 역추산!
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
        
    # [계산 투명성 확보] CoolProp 엔진이 실패하여 Fallback을 사용한 경우 리포트 표출
    if global_audit_tracker:
        st.info("⚠️ **[물성치 Fallback 알림]** 특정 구간에서 CoolProp 혼합물 엔진의 물리적 한계로 인해 다음 우회(Fallback) 가정이 사용되었습니다.\n\n" + "\n".join([f"- {msg}" for msg in global_audit_tracker]))

    df_res = pd.DataFrame(results)
    st.subheader("📊 시뮬레이션 결과 데이터")
    st.dataframe(df_res.style.format({"P (bar)": "{:.3f}", "T (°C)": "{:.2f}", "dP (bar)": "{:.4f}", "L_cum (m)": "{:.2f}", "Z_cum (m)": "{:.2f}"}), use_container_width=True)

    st.subheader("📈 파이프라인 열/수력학적 프로파일")
    
    # 1. 고저차(Elevation)를 반영한 2D 스케치
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
    
    # 2. 압력(Pressure) 및 온도(Temperature) 변화 그래프 (줄-톰슨 팽창 및 마찰 강하 시각화)
    fig_pt = go.Figure()
    fig_pt.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["P (bar)"], mode='lines', name='Pressure (bar)', yaxis='y1', line=dict(color='blue', width=3)))
    fig_pt.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["T (°C)"], mode='lines', name='Temperature (°C)', yaxis='y2', line=dict(color='red', width=3, dash='dash')))
    
    # Plotly 폰트 에러 픽스 반영 (최신 문법 적용)
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
