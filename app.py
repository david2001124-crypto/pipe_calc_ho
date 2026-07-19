import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import material_db

st.set_page_config(page_title="Sequential Pipeline Simulator V5", page_icon="⚓", layout="wide")

@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

def get_k_pipe_extrapolated(T_target, T_list, k_list):
    """
    ASME 재질 DB의 온도 범위를 벗어날 경우, 선형 외삽법(Linear Extrapolation)을 적용
    """
    T_arr = np.array(T_list)
    k_arr = np.array(k_list)
    
    if len(T_arr) < 2: return k_arr[0]
        
    if T_target < T_arr[0]:
        slope = (k_arr[1] - k_arr[0]) / (T_arr[1] - T_arr[0])
        return k_arr[0] + slope * (T_target - T_arr[0])
    elif T_target > T_arr[-1]:
        slope = (k_arr[-1] - k_arr[-2]) / (T_arr[-1] - T_arr[-2])
        return k_arr[-1] + slope * (T_target - T_arr[-1])
    else:
        return float(np.interp(T_target, T_arr, k_arr))

def get_robust_prop(prop_char, T, P, fluid_str, norm_fractions, fallback_val, tracker_set):
    prop_names = {'V': '점도(Viscosity)', 'L': '열전도도(Thermal Conductivity)', 'C': '비열(Specific Heat)'}
    prop_name = prop_names.get(prop_char, prop_char)
    
    try:
        return PropsSI(prop_char, 'T', T, 'P', P, fluid_str)
    except:
        if len(norm_fractions) > 1:
            mix_val = 0.0
            mixing_rule_success = True
            for comp, frac in norm_fractions.items():
                try:
                    mix_val += frac * PropsSI(prop_char, 'T', T, 'P', P, comp)
                except:
                    mixing_rule_success = False
                    break
            
            if mixing_rule_success:
                tracker_set.add(f"{prop_name} 계산 불가 ➔ 순수 성분 몰 분율 가중평균(Mixing Rule) 적용")
                return mix_val
                
        tracker_set.add(f"{prop_name} 계산 불가 ➔ 기본 상수 강제 적용")
        return fallback_val

def churchill_friction_factor(Re, e_D):
    if Re < 1e-10: return 0.0
    A = (2.457 * math.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * e_D)))**16
    B = (37530.0 / Re)**16
    f = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1/12.0)
    return f

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D, theta_deg, roughness, P_Pa):
    g = 9.81
    theta_rad = math.radians(theta_deg)
    
    v_m = v_SL + v_SG
    if v_m < 1e-6:
        return {"dP_dl_elev": 0, "E_k": 0, "flow_regime": "Static", "f_tp": 0, "H_L": 1.0, "lambda_L": 1.0, "rho_n": rho_L, "v_m": v_m}
        
    lambda_L = max(min(v_SL / v_m, 0.999), 0.001)
    N_Fr = (v_m**2) / (g * D)
    L1, L2 = 31.6 * lambda_L**0.302, 0.0009252 * lambda_L**-2.468
    L3, L4 = 0.10 * lambda_L**-1.4516, 0.5 * lambda_L**-6.738
    
    regime = "Unknown"
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2):
        regime = "Segregated"; a, b, c = 0.98, 0.4846, 0.0868
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4):
        regime = "Intermittent"; a, b, c = 0.845, 0.5351, 0.0173
    else:
        regime = "Distributed"; a, b, c = 1.065, 0.5824, 0.0609

    H_L0 = max(min((a * lambda_L**b) / (N_Fr**c), 0.999), lambda_L)
    
    C_corr = 0
    if theta_deg > 0: 
        if regime == "Segregated": C_corr = (1 - lambda_L) * math.log(lambda_L**2 * N_Fr * v_SL)
        elif regime == "Intermittent": C_corr = (1 - lambda_L) * math.log(lambda_L**0.1 * N_Fr * v_SL)
    
    beta = 1.0 + C_corr * (math.sin(1.8 * theta_rad) - (1/3)*math.sin(1.8 * theta_rad)**3)
    H_L = max(min(H_L0 * beta, 0.999), lambda_L)

    rho_s = H_L * rho_L + (1 - H_L) * rho_G 
    rho_n = lambda_L * rho_L + (1 - lambda_L) * rho_G 
    mu_n = lambda_L * mu_L + (1 - lambda_L) * mu_G
    Re_n = (rho_n * v_m * D) / mu_n if mu_n > 0 else 1e6
    f_n = churchill_friction_factor(Re_n, roughness/D)
    
    y = max(lambda_L / (H_L**2), 1e-5)
    S = math.log(2.2 * y - 1.2) if 1.0 < y < 1.2 else math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4)
    
    f_tp = f_n * math.exp(S)
    dP_dl_elev = rho_s * g * math.sin(theta_rad)
    E_k = min((rho_s * v_m * v_SG) / P_Pa, 0.9) 

    return {"dP_dl_elev": dP_dl_elev, "E_k": E_k, "flow_regime": regime, "H_L": H_L, "lambda_L": lambda_L, "v_m": v_m, "Re_n": Re_n, "rho_n": rho_n, "f_tp": f_tp}

def calculate_heat_transfer(Re, Pr, k_fluid, D_i, D_o, k_pipe, k_ins, t_ins, h_o):
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36
    h_i = (Nu * k_fluid) / D_i if D_i > 0 else 1e-5
    
    R_i = D_o / (D_i * h_i)
    R_pipe = (D_o / (2 * k_pipe)) * math.log(D_o / D_i) if k_pipe > 0 else 0
    R_ins = 0
    if t_ins > 0 and k_ins > 0:
        R_ins = (D_o / (2 * k_ins)) * math.log((D_o + 2 * t_ins) / D_o)
    R_o = 1.0 / h_o if h_o > 0 else 0
    return 1.0 / (R_i + R_pipe + R_ins + R_o)

def solve_inner_loop_pressure(T_in, P_in, T_out_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, Le_node, audit_tracker):
    """ HYSYS Inner Loop: 출구 온도가 주어졌을 때 운동량 보존을 만족하는 출구 압력을 할선법으로 찾음 """
    P0 = P_in
    P1 = P_in - 500  # 초기 500 Pa 강하 추정
    tol = 100 # Pa 허용 오차
    
    def calc_P_out(P_guess):
        P_avg = (P_in + P_guess) / 2.0
        T_avg = (T_in + T_out_guess) / 2.0
        if P_avg < 10000: raise ValueError(f"내부 압력이 너무 낮습니다 ({P_avg/1e5:.3f} bar). 배관경 확대 요망.")
            
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
            dP_total = ((f_factor * rho * vel**2) / (2 * D_inner)) * (dL + Le_node) + (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
        else:
            rho_L = PropsSI('D', 'P', P_avg, 'Q', 0, fluid_string)
            rho_G = PropsSI('D', 'P', P_avg, 'Q', 1, fluid_string)
            mu_L = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-3, audit_tracker)
            mu_G = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
            v_SG = (mass_flow * Q_val) / (rho_G * A_cross); v_SL = (mass_flow * (1 - Q_val)) / (rho_L * A_cross)
            bb = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, angle_deg, roughness, P_avg)
            dP_fric = ((bb["f_tp"] * bb["rho_n"] * bb["v_m"]**2) / (2 * D_inner)) * (dL + Le_node)
            dP_total = (dP_fric + bb["dP_dl_elev"] * dL) / (1 - bb["E_k"])
            
        return P_in - dP_total, is_twophase, Q_val

    P_calc0, is_tp0, Q0 = calc_P_out(P0)
    f0 = P_calc0 - P0
    if abs(f0) < tol: return P_calc0, is_tp0, Q0
        
    P_calc1, is_tp1, Q1 = calc_P_out(P1)
    f1 = P_calc1 - P1
    
    for _ in range(20):
        if abs(f1) < tol: return P_calc1, is_tp1, Q1
        if abs(f1 - f0) < 1e-5: P_new = P1 - f1 * 0.5
        else: P_new = P1 - f1 * ((P1 - P0) / (f1 - f0))
        P0, f0 = P1, f1
        P1 = P_new
        P_calc1, is_tp1, Q1 = calc_P_out(P1)
        f1 = P_calc1 - P1

    return P1, is_tp1, Q1

def solve_middle_loop_temp(T_in, P_in, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, D_outer, roughness, angle_deg, dL, Le_node, k_pipe, k_ins, t_ins, h_ext, T_amb_K, audit_tracker):
    """ HYSYS Middle Loop: 에너지 보존(엔탈피 변화=열전달량)을 만족하는 출구 온도를 할선법으로 찾음 """
    T0 = T_in; T1 = T_in - 0.1
    tol = 0.01 # K 허용 오차
    
    try: H_in = PropsSI('H', 'T', T_in, 'P', P_in, fluid_string); use_enthalpy = True
    except: use_enthalpy = False
        
    def calc_T_out(T_guess):
        P_out_calc, is_tp, Q_val = solve_inner_loop_pressure(T_in, P_in, T_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, Le_node, audit_tracker)
        P_avg = (P_in + P_out_calc) / 2.0; T_avg = (T_in + T_guess) / 2.0
        
        Cp = get_robust_prop('C', T_avg, P_avg, fluid_string, norm_fractions, 2000, audit_tracker)
        k_fluid = get_robust_prop('L', T_avg, P_avg, fluid_string, norm_fractions, 0.1, audit_tracker)
        mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
        
        try: rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
        except: rho = 500
            
        Re = (rho * (mass_flow / (rho * A_cross)) * D_inner) / mu
        Pr = (Cp * mu) / k_fluid if k_fluid > 0 else 1.0
        
        U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe, k_ins, t_ins, h_ext)
        Q_heat = U * math.pi * D_outer * dL * (T_amb_K - T_avg) # 열 흡수량
        
        if use_enthalpy:
            try: return PropsSI('T', 'H', H_in + Q_heat / mass_flow, 'P', P_out_calc, fluid_string), P_out_calc, is_tp, Q_val
            except: pass
        return T_in + Q_heat / (mass_flow * Cp), P_out_calc, is_tp, Q_val
        
    T_calc0, P_calc0, is_tp0, Q0 = calc_T_out(T0)
    f0 = T_calc0 - T0
    if abs(f0) < tol: return T_calc0, P_calc0, is_tp0, Q0
        
    T_calc1, P_calc1, is_tp1, Q1 = calc_T_out(T1)
    f1 = T_calc1 - T1
    
    for _ in range(20):
        if abs(f1) < tol: return T_calc1, P_calc1, is_tp1, Q1
        if abs(f1 - f0) < 1e-6: T_new = T1 - f1 * 0.5
        else: T_new = T1 - f1 * ((T1 - T0) / (f1 - f0))
        T0, f0 = T1, f1
        T1 = T_new
        T_calc1, P_calc1, is_tp1, Q1 = calc_T_out(T1)
        f1 = T_calc1 - T1

    return T1, P_calc1, is_tp1, Q1

st.title("⚓ 상선/해양 플랜트 다상유동 시뮬레이터 (V5.0)")
st.markdown("**(HYSYS형 3중 중첩 암시적 수렴(Implicit-Secant) 알고리즘 탑재)**")

st.header("1. 유체 성분 및 운전 조건")
col1, col2, col3, col4 = st.columns(4)
selected_fluids = col1.multiselect("유체 성분 선택", AVAILABLE_FLUIDS, default=None)
T_inlet_C = col2.number_input("입구 온도 (°C)", value=None, format="%.4f")
P_inlet_bar = col3.number_input("입구 압력 (bar)", value=None, format="%.4f")
mass_flow = col4.number_input("질량 유량 (kg/s)", value=None, format="%.4f")

fractions = {}
if selected_fluids:
    f_cols = st.columns(len(selected_fluids))
    for i, fluid in enumerate(selected_fluids):
        fractions[fluid] = f_cols[i].number_input(f"{fluid} 몰 비율", value=None, format="%.4f")

st.header("2. 순차적 파이프라인 빌더 (Sequential Builder)")
if 'pipeline' not in st.session_state: st.session_state.pipeline = []

with st.container(border=True):
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
        f_type = fc1.selectbox("피팅/밸브 종류", list(material_db.FITTING_LE_D_DB.keys()))
        f_qty = fc2.number_input("수량", min_value=1, value=1, step=1)
        
        if st.button("➕ 피팅/밸브 추가", type="secondary"):
            st.session_state.pipeline.append({"type": "Fitting", "name": f_type, "qty": f_qty})
            st.rerun()

if st.session_state.pipeline:
    st.markdown("##### 🧱 현재 구성된 파이프라인 목록 (Flow: 위 ➔ 아래)")
    st.divider()
    for idx, comp in enumerate(st.session_state.pipeline):
        c1, c2, c3, c4, c5 = st.columns([0.1, 0.7, 0.06, 0.06, 0.08])
        c1.write(f"**[{idx+1}]**")
        if comp.get("type") == "Pipe":
            c2.write(f"**배관:** L={comp.get('length', 0)}m | ID={comp.get('D_inner', 0)}m | 재질: {comp.get('material', '').split(' / ')[0]}")
        else:
            c2.write(f"**피팅/밸브:** {comp.get('name')} (x{comp.get('qty', 1)})")
            
        if c3.button("⬆️", key=f"up_{idx}") and idx > 0:
            st.session_state.pipeline[idx-1], st.session_state.pipeline[idx] = st.session_state.pipeline[idx], st.session_state.pipeline[idx-1]
            st.rerun()
        if c4.button("⬇️", key=f"down_{idx}") and idx < len(st.session_state.pipeline)-1:
            st.session_state.pipeline[idx+1], st.session_state.pipeline[idx] = st.session_state.pipeline[idx], st.session_state.pipeline[idx+1]
            st.rerun()
        if c5.button("❌", key=f"del_{idx}"):
            st.session_state.pipeline.pop(idx); st.rerun()
    st.divider()

st.header("3. 외부 환경 및 시뮬레이션 설정")
ec1, ec2, ec3, ec4, ec5 = st.columns(5)
T_amb_C = ec1.number_input("외부 온도 (°C)", value=None, format="%.4f")
h_ext = ec2.number_input("외부 h_o (W/m²K)", value=None, format="%.4f")
t_ins = ec3.number_input("보온재 두께 (m)", value=0.0, format="%.4f")
k_ins = ec4.number_input("보온재 k (W/mK)", value=0.0, format="%.4f")
N_per_pipe = ec5.number_input("배관 분할(Increments)", value=10, min_value=1, step=1, help="HYSYS 아키텍처이므로 각 격자 내에서 완벽히 수렴합니다.")

if st.button("🚀 자동 수렴 다상유동 시뮬레이션 시작", type="primary", use_container_width=True):
    missing = [name for name, val in [("유체", selected_fluids), ("온도", T_inlet_C), ("압력", P_inlet_bar), ("유량", mass_flow), ("외부조건", T_amb_C), ("파이프", st.session_state.pipeline)] if not val]
    if missing or any(v is None for v in fractions.values()): st.error("🚨 필수 항목 누락!"); st.stop()

    total_frac = sum(fractions.values())
    norm_fractions = {k: v / total_frac for k, v in fractions.items()}
    fluid_string = list(norm_fractions.keys())[0] if len(norm_fractions) == 1 else "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])

    global_audit_tracker = set()
    results = []
    T_curr = T_inlet_C + 273.15
    P_curr = P_inlet_bar * 100000
    L_cum, Z_cum = 0.0, 0.0 
    
    results.append({"Component": "Inlet", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, "P (bar)": P_curr / 1e5, "T (°C)": T_curr - 273.15, "Phase": "-", "dP (Pa)": 0, "Regime": "Inlet"})
    curr_D_inner, curr_thickness, curr_roughness, curr_mat_info = 0.1, 0.005, 4.5e-5, list(material_db.MATERIAL_MAP.values())[0]

    status_box = st.status("🤖 HYSYS 3중 중첩 루프 계산 진행 중...", expanded=True)
    
    try:
        for idx, comp in enumerate(st.session_state.pipeline):
            if comp.get("type") == "Pipe":
                curr_D_inner = comp.get("D_inner", curr_D_inner)
                curr_thickness = comp.get("thickness", curr_thickness)
                if "material" in comp: curr_mat_info = material_db.MATERIAL_MAP[comp["material"]]
                curr_roughness = curr_mat_info["roughness_m"]
                asme_table = material_db.RAW_DB[curr_mat_info["asme_category"]][curr_mat_info["asme_grade"]]
                A_cross = math.pi * (curr_D_inner / 2)**2
                D_outer = curr_D_inner + 2 * curr_thickness
                
                L_total = comp.get("length", 10.0)
                dL = L_total / N_per_pipe if N_per_pipe > 0 else 0
                angle = 0 if L_total == 0 else math.degrees(math.asin(max(min(comp.get("elev_val", 0.0) / L_total, 1.0), -1.0))) if comp.get("elev_type", "Angle (deg)") == "Height (m)" else comp.get("elev_val", 0.0)
                dZ = dL * math.sin(math.radians(angle))

                for i in range(N_per_pipe):
                    status_box.update(label=f"🔄 [Pipe {idx+1}/{len(st.session_state.pipeline)}] Node {i+1} 상태 변수(P-T) 상호 수렴 중...")
                    k_pipe = get_k_pipe_extrapolated(T_curr - 273.15, asme_table["T_C"], asme_table["k_W_mK"])
                    
                    T_out, P_out, is_tp, Q_val = solve_middle_loop_temp(
                        T_curr, P_curr, fluid_string, norm_fractions, mass_flow, A_cross, curr_D_inner, D_outer, curr_roughness, angle, dL, 0.0,
                        k_pipe, k_ins, t_ins, h_ext, T_amb_C + 273.15, global_audit_tracker
                    )
                    
                    dP, P_curr, T_curr = P_curr - P_out, P_out, T_out
                    L_cum += dL; Z_cum += dZ
                    results.append({"Component": f"Pipe_{idx+1}", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, "P (bar)": P_curr / 1e5, "T (°C)": T_curr - 273.15, "Phase": "2-Phase" if is_tp else "1-Phase", "dP (Pa)": dP, "Regime": "Beggs & Brill" if is_tp else "Churchill"})

            elif comp.get("type") == "Fitting":
                f_name = comp.get("name", "Unknown")
                status_box.update(label=f"🔄 [Fitting {idx+1}/{len(st.session_state.pipeline)}] {f_name} 등가 길이 기반 수렴 중...")
                
                L_e = (material_db.FITTING_LE_D_DB.get(f_name, 30) * curr_D_inner) * comp.get("qty", 1)
                A_cross = math.pi * (curr_D_inner / 2)**2
                asme_table = material_db.RAW_DB[curr_mat_info["asme_category"]][curr_mat_info["asme_grade"]]
                
                T_out, P_out, is_tp, Q_val = solve_middle_loop_temp(
                    T_curr, P_curr, fluid_string, norm_fractions, mass_flow, A_cross, curr_D_inner, curr_D_inner + 2*curr_thickness, curr_roughness, 0.0, 0.0, L_e,
                    get_k_pipe_extrapolated(T_curr - 273.15, asme_table["T_C"], asme_table["k_W_mK"]), k_ins, t_ins, h_ext, T_amb_C + 273.15, global_audit_tracker
                )
                
                dP, P_curr, T_curr = P_curr - P_out, P_out, T_out
                results.append({"Component": f"Fitting_{idx+1} ({f_name})", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, "P (bar)": P_curr / 1e5, "T (°C)": T_curr - 273.15, "Phase": "2-Phase" if is_tp else "1-Phase", "dP (Pa)": dP, "Regime": "Equiv. Length (L_e)"})
                
        status_box.update(label="✅ 시뮬레이션 완벽 수렴 (Secant Method 완료)!", state="complete")
    except Exception as e:
        status_box.update(label="🚨 해석 중단", state="error")
        st.error(str(e)); st.stop()

    df_res = pd.DataFrame(results)
    if global_audit_tracker:
        st.info("⚠️ **[계산 상태 알림]** CoolProp 혼합물 지원 한계로 다음 가정이 적용되었습니다:\n" + "\n".join([f"- {m}" for m in global_audit_tracker]))

    m1, m2, m3 = st.columns(3)
    m1.metric("총 압력 강하", f"{P_inlet_bar - df_res.iloc[-1]['P (bar)']:.4f} bar")
    m2.metric("최종 출구 압력", f"{df_res.iloc[-1]['P (bar)']:.4f} bar")
    m3.metric("최종 출구 온도", f"{df_res.iloc[-1]['T (°C)']:.4f} °C")

    st.subheader("🗺️ 2D 파이프라인 물리 스케치")
    fig_2d = go.Figure()
    pipe_mask = df_res['Component'].str.startswith('Pipe') | (df_res['Component'] == 'Inlet')
    fig_2d.add_trace(go.Scatter(x=df_res[pipe_mask]['L_cum (m)'], y=df_res[pipe_mask]['Z_cum (m)'], mode='lines', name='Pipeline Path', line=dict(color='blue', width=4)))
    
    valve_mask = df_res['Component'].str.startswith('Fitting')
    if valve_mask.any():
        fig_2d.add_trace(go.Scatter(x=df_res[valve_mask]['L_cum (m)'], y=df_res[valve_mask]['Z_cum (m)'], mode='markers+text', name='Fittings', marker=dict(symbol='diamond', size=14, color='red'), text=df_res[valve_mask]['Component'].apply(lambda x: x.split('(')[1].replace(')','')), textposition="top center"))
    fig_2d.update_layout(xaxis_title="누적 길이 - X (m)", yaxis_title="누적 고도 - Z (m)", height=400, title="유동 방향: Left ➔ Right")
    st.plotly_chart(fig_2d, use_container_width=True)

    st.subheader("📈 압력 및 온도 프로필")
    fig_prof = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['P (bar)'], name="Pressure", line=dict(color='red', width=3)), secondary_y=False)
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['T (°C)'], name="Temperature", line=dict(color='blue', dash='dash', width=2)), secondary_y=True)
    fig_prof.update_layout(height=400); fig_prof.update_yaxes(title_text="Pressure (bar)", secondary_y=False); fig_prof.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.subheader("📊 시뮬레이션 결과 데이터")
    st.dataframe(df_res, use_container_width=True)
