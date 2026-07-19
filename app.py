import streamlit as st
from CoolProp.CoolProp import PropsSI, PhaseSI
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
import material_db

st.set_page_config(page_title="해양 파이프라인 시뮬레이터", layout="wide")

# ==========================================
# 1. 공학/물리 계산 헬퍼 함수
# ==========================================

def churchill_friction_factor(Re, ed):
    """ Churchill (1977) 마찰계수: 층류, 전이, 난류 전 구간 포괄 """
    if Re <= 0: return 0.01
    A = (-2.457 * math.log((7.0 / Re)**0.9 + 0.27 * ed))**16
    B = (37530.0 / Re)**16
    f = 8 * ((8 / Re)**12 + 1 / (A + B)**1.5)**(1/12)
    return f

def calculate_fT_hysys(roughness, D_inner):
    """ HYSYS 기준: 해당 배관 조도에서 레이놀즈 수를 극단적으로 높여 Fully Turbulent 마찰 계수(fT) 도출 """
    Re_test = 1e6
    f_old = 0.0
    for _ in range(100):
        f_new = churchill_friction_factor(Re_test, roughness / D_inner)
        if abs(f_new - f_old) < 1e-7:
            return f_new
        f_old = f_new
        Re_test *= 10
    return f_new

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, angle_deg, roughness, P_Pa):
    """ Beggs and Brill 2상 유동(Two-phase flow) 압력 강하 모델 """
    v_m = v_SL + v_SG
    if v_m <= 0: v_m = 1e-6
    lambda_L = v_SL / v_m
    N_Fr = (v_m**2) / (9.81 * D_inner)
    
    L1 = 316 * lambda_L**0.302; L2 = 0.000925 * lambda_L**-2.4684
    L3 = 0.1 * lambda_L**-1.4516; L4 = 0.5 * lambda_L**-6.738
    
    # 맵 기반 유동 양식 판별
    regime = "Transition"
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2): regime = "Segregated"
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4): regime = "Intermittent"
    elif (lambda_L < 0.4 and N_Fr >= L1) or (lambda_L >= 0.4 and N_Fr > L4): regime = "Distributed"
    
    if regime == "Segregated": a, b, c = 0.98, 0.4846, 0.0868
    elif regime == "Intermittent": a, b, c = 0.845, 0.5351, 0.0173
    elif regime == "Distributed": a, b, c = 1.065, 0.5824, 0.0609
    else: a, b, c = 0.9125, 0.50985, 0.05205

    H_L_0 = a * lambda_L**b / N_Fr**c
    if H_L_0 < lambda_L: H_L_0 = lambda_L
    if H_L_0 > 1.0: H_L_0 = 1.0

    C_val = (1 - lambda_L) * math.log(max(lambda_L, 1e-5) * 0.01**0.05 * 1e5**0.1 * 1e5**0.1) # 단순화 보정
    if regime == "Segregated": beta = max(0, (1 - lambda_L) * math.log(max(1e-5, C_val)))
    else: beta = 0
    
    H_L = H_L_0 * (1 + beta * math.sin(math.radians(angle_deg)))
    if H_L < 0: H_L = 0
    elif H_L > 1: H_L = 1
    
    rho_n = rho_L * lambda_L + rho_G * (1 - lambda_L)
    rho_s = rho_L * H_L + rho_G * (1 - H_L)
    mu_n = mu_L * lambda_L + mu_G * (1 - lambda_L)
    Re_n = (rho_n * v_m * D_inner) / mu_n
    f_n = churchill_friction_factor(Re_n, roughness/D_inner)
    
    y = lambda_L / H_L**2 if H_L > 0 else 1.0
    S = math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4) if 1 < y < 1.2 else math.log(y)
    f_tp = f_n * math.exp(S)
    
    dP_dl_elev = rho_s * 9.81 * math.sin(math.radians(angle_deg))
    E_k = (v_m * v_SG * rho_n) / P_Pa if P_Pa > 0 else 0
    
    return {"f_tp": f_tp, "rho_n": rho_n, "v_m": v_m, "dP_dl_elev": dP_dl_elev, "E_k": E_k, "flow_regime": regime, "lambda_L": lambda_L, "Re_n": Re_n}

def get_k_pipe_extrapolated(T_C, T_arr, k_arr):
    if T_C >= T_arr[0] and T_C <= T_arr[-1]:
        return np.interp(T_C, T_arr, k_arr)
    elif T_C < T_arr[0]:
        slope = (k_arr[1] - k_arr[0]) / (T_arr[1] - T_arr[0])
        return k_arr[0] + slope * (T_C - T_arr[0])
    else:
        slope = (k_arr[-1] - k_arr[-2]) / (T_arr[-1] - T_arr[-2])
        return k_arr[-1] + slope * (T_C - T_arr[-1])

def get_robust_prop(prop, T, P, fluid_string, fractions, default_val, tracker):
    try:
        return PropsSI(prop, 'T', T, 'P', P, fluid_string)
    except Exception:
        fluids = [f.split('[')[0] for f in fluid_string.split('&')]
        if len(fluids) == len(fractions):
            val_mix = 0.0
            try:
                for i, f in enumerate(fluids):
                    val_mix += fractions[i] * PropsSI(prop, 'T', T, 'P', P, f)
                tracker.add(f"'{prop}' 물성치 계산 실패 ➔ Leduc 가중 평균(Mixing Rule)으로 우회 계산됨.")
                return val_mix
            except: pass
        tracker.add(f"'{prop}' 물성치 계산 실패 ➔ 설정된 기본 상수({default_val})로 강제 적용됨.")
        return default_val

def calculate_heat_transfer(Re, Pr, k_fluid, D_in, D_out, k_pipe, k_ins, t_ins, h_ext):
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36
    h_in = (Nu * k_fluid) / D_in if D_in > 0 else 1000
    R_conv_in = 1.0 / (h_in * math.pi * D_in)
    R_cond_pipe = math.log(D_out / D_in) / (2 * math.pi * k_pipe) if D_out > D_in else 0
    D_ins_out = D_out + 2 * t_ins
    R_cond_ins = math.log(D_ins_out / D_out) / (2 * math.pi * k_ins) if t_ins > 0 else 0
    R_conv_out = 1.0 / (h_ext * math.pi * D_ins_out)
    R_total = R_conv_in + R_cond_pipe + R_cond_ins + R_conv_out
    return 1.0 / (R_total * math.pi * D_ins_out) if R_total > 0 else 0

# ==========================================
# 2. 핵심 물리 엔진 (Nested Loop)
# ==========================================

def solve_inner_loop_pressure(T_in, P_in, T_out_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, audit_tracker):
    """ HYSYS Inner Loop: 출구 온도가 주어졌을 때 운동량 보존(Beggs&Brill 등)을 만족하는 출구 압력을 할선법(Secant)으로 도출 """
    P0 = P_in
    P1 = P_in - 500
    tol = 100
    
    def calc_P_out(P_guess):
        P_avg = (P_in + P_guess) / 2.0
        T_avg = (T_in + T_out_guess) / 2.0
        if P_avg < 10000: raise ValueError(f"내부 압력이 너무 낮습니다 ({P_avg/1e5:.3f} bar).")
            
        try: phase_raw = PhaseSI('T', T_avg, 'P', P_avg, fluid_string)
        except: phase_raw = "unknown"
        
        Q_val = -1.0; is_twophase = False
        if phase_raw == 'twophase':
            is_twophase = True
            try: Q_val = PropsSI('Q', 'T', T_avg, 'P', P_avg, fluid_string)
            except: is_twophase = False
            
        if not is_twophase:
            rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
            mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
            vel = mass_flow / (rho * A_cross)
            Re = (rho * vel * D_inner) / mu
            f_factor = churchill_friction_factor(Re, roughness/D_inner)
            dP_total = ((f_factor * rho * vel**2) / (2 * D_inner)) * dL + (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
            regime = "1-Phase"
        else:
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

    P_calc0, is_tp0, Q0, reg0 = calc_P_out(P0)
    f0 = P_calc0 - P0
    if abs(f0) < tol: return P_calc0, is_tp0, Q0, reg0
        
    P_calc1, is_tp1, Q1, reg1 = calc_P_out(P1)
    f1 = P_calc1 - P1
    
    for _ in range(20):
        if abs(f1) < tol: return P_calc1, is_tp1, Q1, reg1
        if abs(f1 - f0) < 1e-5: P_new = P1 - f1 * 0.5
        else: P_new = P1 - f1 * ((P1 - P0) / (f1 - f0))
        
        P0, f0 = P1, f1
        P1 = P_new
        P_calc1, is_tp1, Q1, reg1 = calc_P_out(P1)
        f1 = P_calc1 - P1

    return P1, is_tp1, Q1, reg1

def solve_middle_loop_temp(T_in, P_in, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, D_outer, roughness, angle_deg, dL, k_pipe, k_ins, t_ins, h_ext, T_amb_K, audit_tracker):
    """ HYSYS Middle Loop: 에너지 보존을 만족하는 출구 온도를 할선법(Secant)으로 도출 """
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
        
        T0, f0 = T1, f1
        T1 = T_new
        T_calc1, P_calc1, is_tp1, Q1, reg1 = calc_T_out(T1)
        f1 = T_calc1 - T1

    return T1, P_calc1, is_tp1, Q1, reg1

# ==========================================
# 3. UI 및 상태 관리
# ==========================================

st.title("⚓ 해양/플랜트 다상 유동 파이프라인 시뮬레이터 (V6.0)")
st.markdown("**(HYSYS 3-Nested-Loop, PH-Flash, 디커플링 물리 엔진 적용)**")

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
    f_type = fc1.selectbox("피팅/밸브 종류", list(material_db.HYSYS_FITTING_DB.keys()))
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
N_per_pipe = col_env3.number_input("배관당 분할 격자 수 (N)", min_value=1, value=10, step=1, help="내부 루프 수렴으로 인해 적은 N으로도 높은 정확도를 확보합니다.")

col_env4, col_env5 = st.columns(2)
t_ins = col_env4.number_input("보온재 두께 (m)", min_value=0.0, value=0.0, format="%.4f")
k_ins = col_env5.number_input("보온재 열전도도 (W/m·K)", value=0.035, format="%.4f")

# ==========================================
# 4. 시뮬레이션 실행 및 시각화
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
    status_box = st.status("🤖 3-Nested-Loop 및 피팅 디커플링 해석 진행 중...", expanded=True)
    
    try:
        for comp_idx, comp in enumerate(st.session_state.pipeline):
            if comp.get("type") == "Pipe":
                # [A] 직관(Pipe) 물리 해석: 3중 루프 및 Beggs & Brill
                status_box.update(label=f"🔄 [Pipe {comp_idx+1}/{len(st.session_state.pipeline)}] 3중 루프 수렴(Outer-Middle-Inner) 계산 중...")
                
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

            elif comp.get("type") == "Fitting":
                # [B] 피팅/밸브(Fitting) 물리 해석: Crane TP-410, Chisholm B, PH-Flash 열역학
                f_name = comp.get("name", "Unknown")
                qty = comp.get("qty", 1)
                status_box.update(label=f"🔄 [Fitting {comp_idx+1}/{len(st.session_state.pipeline)}] {f_name} (x{qty}) 국부 저항 계산 중...")
                
                if P_current_Pa < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 통과 전 압력이 {P_current_Pa/1e5:.3f} bar로 너무 낮습니다.")

                A_cross = math.pi * (curr_D_inner / 2)**2
                fit_data = material_db.HYSYS_FITTING_DB.get(f_name, {"A": 0.0, "B": 30, "Chisholm_B": 1.5})
                
                # 1. 속도 수두 인자 K 계산 (Crane TP-410 방식)
                f_T = calculate_fT_hysys(curr_roughness, curr_D_inner)
                K_factor = (fit_data["A"] + fit_data["B"] * f_T) * qty
                
                # 2. 유동 상태 판별
                try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                except: phase_raw = "unknown"
                is_twophase = (phase_raw == 'twophase')
                
                # 3. 국부 압력 강하(dP) 계산
                if not is_twophase:
                    rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    vel = mass_flow / (rho * A_cross)
                    dP_fit = K_factor * rho * (vel**2) / 2.0
                    regime_label = "Fitting (1-Phase)"
                else:
                    Q_val = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    G_mass_flux = mass_flow / A_cross
                    
                    dP_LO = K_factor * (G_mass_flux**2) / (2 * rho_L) # Liquid-Only Pressure Drop
                    # Chisholm Two-Phase Multiplier
                    phi_LO2 = 1 + (rho_L / rho_G - 1) * (fit_data["Chisholm_B"] * Q_val * (1 - Q_val) + Q_val**2)
                    dP_fit = dP_LO * phi_LO2
                    regime_label = "Fitting (2-Phase)"
                
                P_out_fit = P_current_Pa - dP_fit
                if P_out_fit < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 밸브 통과 후 압력이 진공 수준({P_out_fit/1e5:.3f} bar)입니다. 밸브를 열거나 유량을 줄이세요.")
                
                # 4. 온도 업데이트: PH-Flash (등엔탈피 팽창, Joule-Thomson 효과 모사)
                try:
                    H_in = PropsSI('H', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    T_out_fit = PropsSI('T', 'H', H_in, 'P', P_out_fit, fluid_string)
                except:
                    T_out_fit = T_current_K # Fail-safe fallback
                
                P_current_Pa = P_out_fit
                T_current_K = T_out_fit
                
                results.append({
                    "Component": f"Fitting_{comp_idx+1} ({f_name})", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, 
                    "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, 
                    "Phase": "2-Phase" if is_twophase else "1-Phase", "dP (bar)": dP_fit / 1e5, 
                    "Regime": regime_label
                })

        status_box.update(label="✅ 물리 모델 해석 완료!", state="complete")
        
    except ValueError as ve:
        status_box.update(label="🚨 시뮬레이션 중단", state="error")
        st.error(f"{ve}")
        st.stop()
    except Exception as e:
        status_box.update(label="🚨 오류 발생", state="error")
        st.error(f"예상치 못한 에러: {e}")
        st.stop()
        
    if global_audit_tracker:
        st.info("⚠️ **[물성치 Fallback 알림]** CoolProp 혼합물 엔진 한계로 일부 구간에서 다음 가정이 사용되었습니다.\n\n" + "\n".join([f"- {msg}" for msg in global_audit_tracker]))

    df_res = pd.DataFrame(results)
    st.subheader("📊 시뮬레이션 결과 데이터")
    st.dataframe(df_res.style.format({"P (bar)": "{:.3f}", "T (°C)": "{:.2f}", "dP (bar)": "{:.4f}", "L_cum (m)": "{:.2f}", "Z_cum (m)": "{:.2f}"}), use_container_width=True)

    st.subheader("📈 파이프라인 프로파일")
    
    # 2D 형상 시각화
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["Z_cum (m)"], mode='lines', line=dict(color='gray', width=4), name='Pipeline'))
    
    fittings_df = df_res[df_res['Component'].str.contains("Fitting")]
    if not fittings_df.empty:
        fig_2d.add_trace(go.Scatter(
            x=fittings_df["L_cum (m)"], y=fittings_df["Z_cum (m)"],
            mode='markers', marker=dict(color='red', size=12, symbol='diamond'),
            name='Valve/Fitting', hovertext=fittings_df['Component']
        ))
    
    fig_2d.update_layout(title="파이프라인 2D 스케치 (Side View)", xaxis_title="누적 길이 (m)", yaxis_title="고도 (m)", showlegend=True)
    
    fig_pt = go.Figure()
    fig_pt.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["P (bar)"], mode='lines', name='Pressure (bar)', yaxis='y1', line=dict(color='blue')))
    fig_pt.add_trace(go.Scatter(x=df_res["L_cum (m)"], y=df_res["T (°C)"], mode='lines', name='Temperature (°C)', yaxis='y2', line=dict(color='red', dash='dash')))
    
    fig_pt.update_layout(
        title="압력 및 온도 프로파일 (Valve 구간 압력 강하 주목)", xaxis_title="누적 길이 (m)",
        yaxis=dict(title="압력 (bar)", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="온도 (°C)", titlefont=dict(color="red"), tickfont=dict(color="red"), overlaying="y", side="right"),
        hovermode="x unified"
    )
    
    tc1, tc2 = st.columns(2)
    tc1.plotly_chart(fig_2d, use_container_width=True)
    tc2.plotly_chart(fig_pt, use_container_width=True)
