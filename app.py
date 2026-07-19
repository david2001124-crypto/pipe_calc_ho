import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import material_db

st.set_page_config(page_title="Sequential Pipeline Simulator V6", page_icon="⚓", layout="wide")

@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

def get_k_pipe_extrapolated(T_target, T_list, k_list):
    """ ASME 재질 DB 온도 범위 이탈 시 선형 외삽(Linear Extrapolation) 수행 """
    T_arr = np.array(T_list); k_arr = np.array(k_list)
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
    """ 혼합물 물성치 계산 실패 시 Mixing Rule(가중 평균) 적용 및 기록(Audit Trail) """
    prop_names = {'V': '점도(Viscosity)', 'L': '열전도도(Thermal Conductivity)', 'C': '비열(Specific Heat)'}
    prop_name = prop_names.get(prop_char, prop_char)
    try: return PropsSI(prop_char, 'T', T, 'P', P, fluid_str)
    except:
        if len(norm_fractions) > 1:
            mix_val = 0.0; success = True
            for comp, frac in norm_fractions.items():
                try: mix_val += frac * PropsSI(prop_char, 'T', T, 'P', P, comp)
                except: success = False; break
            if success:
                tracker_set.add(f"{prop_name} 혼합물 계산 실패 ➔ 단일 성분 몰 분율 가중평균(Mixing Rule) 적용")
                return mix_val
        tracker_set.add(f"{prop_name} 계산 완전 실패 ➔ 기본(Default) 상수 강제 적용")
        return fallback_val

def churchill_friction_factor(Re, e_D):
    """ 단상 유동 마찰 계수 계산 (층류~완전 난류 연속 함수) """
    if Re < 1e-10: return 0.0
    A = (2.457 * math.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * e_D)))**16
    B = (37530.0 / Re)**16
    return 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1/12.0)

def calculate_fT_hysys(roughness, D):
    """ HYSYS 매뉴얼 기반: 레이놀즈 수를 무한대로 증가시키며 Churchill 공식을 반복 호출하여 수렴하는 극한값(f_T)을 찾음 """
    e_D = max(roughness / D, 1e-9)
    Re_test = 1e6
    f_old = churchill_friction_factor(Re_test, e_D)
    
    for _ in range(20):
        Re_test *= 10
        f_new = churchill_friction_factor(Re_test, e_D)
        if abs(f_new - f_old) < 1e-8:
            return f_new
        f_old = f_new
    return f_new

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D, theta_deg, roughness, P_Pa):
    """ 파이프 2상 유동(Beggs & Brill 1973) 압력 강하 및 정수두 계산 """
    g = 9.81; theta_rad = math.radians(theta_deg)
    v_m = v_SL + v_SG
    if v_m < 1e-6: return {"dP_dl_elev": rho_L * g * math.sin(theta_rad), "E_k": 0, "flow_regime": "Static", "f_tp": 0.02, "H_L": 1.0, "lambda_L": 1.0, "rho_n": rho_L, "v_m": 0.0}
        
    lambda_L = max(min(v_SL / v_m, 0.999), 0.001)
    N_Fr = (v_m**2) / (g * D)
    L1, L2 = 31.6 * lambda_L**0.302, 0.0009252 * lambda_L**-2.468
    L3, L4 = 0.10 * lambda_L**-1.4516, 0.5 * lambda_L**-6.738
    
    regime = "Distributed"; a, b, c = 1.065, 0.5824, 0.0609
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2): regime = "Segregated"; a, b, c = 0.98, 0.4846, 0.0868
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4): regime = "Intermittent"; a, b, c = 0.845, 0.5351, 0.0173

    H_L0 = max(min((a * lambda_L**b) / (N_Fr**c), 0.999), lambda_L)
    C_corr = 0
    if theta_deg > 0: 
        if regime == "Segregated": C_corr = (1 - lambda_L) * math.log(max(lambda_L**2 * N_Fr * v_SL, 1e-5))
        elif regime == "Intermittent": C_corr = (1 - lambda_L) * math.log(max(lambda_L**0.1 * N_Fr * v_SL, 1e-5))
    
    beta = 1.0 + C_corr * (math.sin(1.8 * theta_rad) - (1/3)*math.sin(1.8 * theta_rad)**3)
    H_L = max(min(H_L0 * beta, 0.999), lambda_L)

    rho_s = H_L * rho_L + (1 - H_L) * rho_G 
    rho_n = lambda_L * rho_L + (1 - lambda_L) * rho_G 
    mu_n = lambda_L * mu_L + (1 - lambda_L) * mu_G
    Re_n = (rho_n * v_m * D) / mu_n if mu_n > 0 else 1e6
    f_n = churchill_friction_factor(Re_n, roughness/D)
    
    y = max(lambda_L / (H_L**2), 1e-5)
    S = math.log(2.2 * y - 1.2) if 1.0 < y < 1.2 else math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4)
    
    return {"dP_dl_elev": rho_s * g * math.sin(theta_rad), "E_k": min((rho_s * v_m * v_SG) / P_Pa, 0.9), "flow_regime": regime, "H_L": H_L, "lambda_L": lambda_L, "v_m": v_m, "Re_n": Re_n, "rho_n": rho_n, "f_tp": f_n * math.exp(S)}

def calculate_heat_transfer(Re, Pr, k_fluid, D_i, D_o, k_pipe, k_ins, t_ins, h_o, is_heating=False):
    """ 1D 원통형 열전달 저항 네트워크 계산 (전체 U-value 산출) """
    n_exp = 0.4 if is_heating else 0.3
    Nu = 0.023 * (Re**0.8) * (Pr**n_exp) if Re > 2300 else 4.36
    h_i = (Nu * k_fluid) / D_i if D_i > 0 else 1e-5
    
    D_ins_outer = D_o + 2 * t_ins
    R_conv_in = 1.0 / (h_i * math.pi * D_i)
    R_cond_pipe = math.log(D_o / D_i) / (2 * math.pi * k_pipe) if k_pipe > 0 else 0
    R_cond_ins = math.log(D_ins_outer / D_o) / (2 * math.pi * k_ins) if (t_ins > 0 and k_ins > 0) else 0
    R_conv_out = 1.0 / (h_o * math.pi * D_ins_outer) if h_o > 0 else 0
    
    return 1.0 / (R_conv_in + R_cond_pipe + R_cond_ins + R_conv_out)

def solve_inner_loop_pressure(T_in, P_in, T_out_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, audit_tracker):
    """ HYSYS Inner Loop: 출구 온도가 주어졌을 때 운동량 보존을 만족하는 출구 압력을 할선법으로 찾음 """
    P0 = P_in; P1 = P_in - 500; tol = 100 
    
    def calc_P_out(P_guess):
        P_avg = (P_in + P_guess) / 2.0; T_avg = (T_in + T_out_guess) / 2.0
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
            # V6: 피팅 등가길이 배제, 순수 배관 마찰만 계산
            dP_total = ((f_factor * rho * vel**2) / (2 * D_inner)) * dL + (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
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

def solve_middle_loop_temp(T_in, P_in, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, D_outer, roughness, angle_deg, dL, k_pipe, k_ins, t_ins, h_ext, T_amb_K, audit_tracker):
    """ HYSYS Middle Loop: 에너지 보존(엔탈피 변화=열전달량)을 만족하는 출구 온도를 할선법으로 찾음 """
    T0 = T_in; T1 = T_in - 0.1; tol = 0.01
    
    try: 
        H_in = PropsSI('H', 'T', T_in, 'P', P_in, fluid_string)
        use_enthalpy = True
    except: 
        use_enthalpy = False
        
    def calc_T_out(T_guess):
        P_out_calc, is_tp, Q_val = solve_inner_loop_pressure(T_in, P_in, T_guess, fluid_string, norm_fractions, mass_flow, A_cross, D_inner, roughness, angle_deg, dL, audit_tracker)
        P_avg = (P_in + P_out_calc) / 2.0; T_avg = (T_in + T_guess) / 2.0
        
        Cp = get_robust_prop('C', T_avg, P_avg, fluid_string, norm_fractions, 2000, audit_tracker)
        k_fluid = get_robust_prop('L', T_avg, P_avg, fluid_string, norm_fractions, 0.1, audit_tracker)
        mu = get_robust_prop('V', T_avg, P_avg, fluid_string, norm_fractions, 1e-5, audit_tracker)
        
        try: rho = PropsSI('D', 'T', T_avg, 'P', P_avg, fluid_string)
        except: rho = 500
            
        Re = (rho * (mass_flow / (rho * A_cross)) * D_inner) / mu
        Pr = (Cp * mu) / k_fluid if k_fluid > 0 else 1.0
        
        U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe, k_ins, t_ins, h_ext)
        Q_heat = U * dL * (T_amb_K - T_avg) 
        
        if use_enthalpy:
            try: 
                return PropsSI('T', 'H', H_in + Q_heat / mass_flow, 'P', P_out_calc, fluid_string), P_out_calc, is_tp, Q_val
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

st.title("⚓ 해양/플랜트 다상 유동 파이프라인 시뮬레이터 (V6.0)")
st.caption("기반 물리 엔진: 3-Nested-Loop, PH-Flash Thermodynamics, Beggs & Brill (Pipe), Crane TP-410 & Chisholm (Fittings)")

st.header("1. 유체 성분 및 운전 조건")
c1, c2, c3, c4 = st.columns(4)
selected_fluids = c1.multiselect("유체 구성 성분 선택 (CoolProp 기반)", AVAILABLE_FLUIDS, default=["Methane", "Ethane"])
fractions = {}
if selected_fluids:
    st.write("성분별 몰 분율 (Mole Fractions) 입력:")
    frac_cols = st.columns(len(selected_fluids))
    for i, f in enumerate(selected_fluids):
        fractions[f] = frac_cols[i].number_input(f"{f} 분율", min_value=0.0, value=1.0/len(selected_fluids), format="%.3f")

T_inlet_C = c2.number_input("입구 온도 (°C)", value=20.0, format="%.2f")
P_inlet_bar = c3.number_input("입구 압력 (bar)", value=10.0, format="%.2f")
mass_flow = c4.number_input("질량 유량 (kg/s)", value=5.0, min_value=0.001, format="%.3f")

st.header("2. 순차적 파이프라인 빌더")
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

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
ec1, ec2, ec3, ec4, ec5 = st.columns(5)
T_amb_C = ec1.number_input("외부 온도 (°C)", value=25.0, format="%.2f")
h_ext = ec2.number_input("외부 열전달계수 (W/m²K)", value=10.0, format="%.2f")
t_ins = ec3.number_input("보온재 두께 (m)", value=0.0, format="%.4f")
k_ins = ec4.number_input("보온재 열전도도 (W/mK)", value=0.0, format="%.4f")
N_per_pipe = ec5.number_input("배관당 내부 분할(Node) 갯수", value=10, min_value=1, step=1, help="Nested Loop 수렴을 통해 각 격자 내 오차를 완벽히 해결하므로 적절한 N 설정")

if st.button("🚀 고급 물리엔진 시뮬레이션 시작", type="primary", use_container_width=True):
    missing = [name for name, val in [("유체", selected_fluids), ("온도", T_inlet_C), ("압력", P_inlet_bar), ("유량", mass_flow), ("외부조건", T_amb_C), ("파이프", st.session_state.pipeline)] if not val]
    if missing or any(v is None for v in fractions.values()): st.error("🚨 필수 항목 누락!"); st.stop()

    total_frac = sum(fractions.values())
    norm_fractions = {k: v / total_frac for k, v in fractions.items()}
    fluid_string = list(norm_fractions.keys())[0] if len(norm_fractions) == 1 else "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])

    global_audit_tracker = set()
    results = []
    
    T_current_K = T_inlet_C + 273.15
    P_current_Pa = P_inlet_bar * 100000
    L_cum, Z_cum = 0.0, 0.0 
    
    results.append({"Component": "Inlet", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, "Phase": "-", "dP (Pa)": 0, "Regime": "Inlet"})
    
    curr_D_inner, curr_thickness, curr_roughness, curr_mat_info = 0.1, 0.005, 4.5e-5, list(material_db.MATERIAL_MAP.values())[0]
    status_box = st.status("🤖 3-Nested-Loop 및 피팅 디커플링 해석 진행 중...", expanded=True)
    
    try:
        for comp_idx, comp in enumerate(st.session_state.pipeline):
            if comp.get("type") == "Pipe":
                # [A] 직관 배관: 3중 중첩 루프 기반 PH-Flash 해석 적용
                curr_D_inner = comp.get("D_inner", curr_D_inner)
                curr_thickness = comp.get("thickness", curr_thickness)
                if "material" in comp: curr_mat_info = material_db.MATERIAL_MAP[comp["material"]]
                curr_roughness = curr_mat_info["roughness_m"]
                asme_table = material_db.RAW_DB[curr_mat_info["asme_category"]][curr_mat_info["asme_grade"]]
                D_outer = curr_D_inner + 2 * curr_thickness
                A_cross = math.pi * (curr_D_inner / 2)**2
                
                length = comp.get("length", 10.0)
                dL = length / N_per_pipe if N_per_pipe > 0 else 0
                angle_deg = 0 if length == 0 else math.degrees(math.asin(max(min(comp.get("elev_val", 0.0) / length, 1.0), -1.0))) if comp.get("elev_type", "Angle (deg)") == "Height (m)" else comp.get("elev_val", 0.0)
                dZ = dL * math.sin(math.radians(angle_deg))

                for i in range(N_per_pipe):
                    status_box.update(label=f"🔄 [Pipe {comp_idx+1}/{len(st.session_state.pipeline)}] Node {i+1} 에너지-운동량 암시적 수렴(Implicit Secant) 중...")
                    k_pipe_current = get_k_pipe_extrapolated(T_current_K - 273.15, asme_table["T_C"], asme_table["k_W_mK"])
                    
                    T_out, P_out, is_tp, Q_val = solve_middle_loop_temp(
                        T_current_K, P_current_Pa, fluid_string, norm_fractions, mass_flow, A_cross, 
                        curr_D_inner, D_outer, curr_roughness, angle_deg, dL, 
                        k_pipe_current, k_ins, t_ins, h_ext, T_amb_C + 273.15, global_audit_tracker
                    )
                    
                    dP, P_current_Pa, T_current_K = P_current_Pa - P_out, P_out, T_out
                    L_cum += dL; Z_cum += dZ
                    results.append({"Component": f"Pipe_{comp_idx+1}", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, "Phase": "2-Phase" if is_tp else "1-Phase", "dP (Pa)": dP, "Regime": "Beggs & Brill" if is_tp else "Churchill"})

            elif comp.get("type") == "Fitting":
                # [B] 피팅/밸브: Crane TP-410 & Chisholm 모델로 독립적 해석 (디커플링)
                f_name = comp.get("name", "Unknown")
                qty = comp.get("qty", 1)
                
                if P_current_Pa < 10000: raise ValueError(f"[{comp_idx+1}번 밸브] 통과 전 압력이 {P_current_Pa/1e5:.3f} bar로 너무 낮습니다.")

                fit_data = material_db.HYSYS_FITTING_DB.get(f_name, {"A": 0.0, "B": 30, "Chisholm_B": 1.5})
                A_vh = fit_data["A"]
                B_ft = fit_data["B"]
                B_param = fit_data["Chisholm_B"]
                
                try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                except: phase_raw = "unknown"
                
                is_twophase = False; Q_val = -1.0
                if phase_raw == 'twophase':
                    is_twophase = True
                    try: Q_val = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    except: is_twophase = False
                
                f_T = calculate_fT_hysys(curr_roughness, curr_D_inner)
                K_factor = (A_vh + B_ft * f_T) * qty
                
                if not is_twophase:
                    rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    A_cross = math.pi * (curr_D_inner / 2)**2
                    vel = mass_flow / (rho * A_cross)
                    dP_fit = K_factor * rho * (vel**2) / 2.0
                else:
                    rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    A_cross = math.pi * (curr_D_inner / 2)**2
                    G_mass_flux = mass_flow / A_cross
                    dP_LO = K_factor * (G_mass_flux**2) / (2.0 * rho_L) 
                    Phi_LO2 = 1.0 + (rho_L / rho_G - 1.0) * (B_param * Q_val * (1.0 - Q_val) + Q_val**2) 
                    dP_fit = dP_LO * Phi_LO2
                
                P_out_fit = P_current_Pa - dP_fit
                
                # PH-Flash Thermodynamics: 피팅 통과 시 등엔탈피(Isoenthalpic) 팽창 가정
                try:
                    H_in = PropsSI('H', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    T_out_fit = PropsSI('T', 'H', H_in, 'P', P_out_fit, fluid_string)
                except:
                    T_out_fit = T_current_K # Fail-safe fallback
                
                P_current_Pa, T_current_K = P_out_fit, T_out_fit
                results.append({"Component": f"Fitting_{comp_idx+1} ({f_name})", "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15, "Phase": "2-Phase" if is_twophase else "1-Phase", "dP (Pa)": dP_fit, "Regime": "Chisholm 2-Phase" if is_twophase else "Crane TP-410"})

        status_box.update(label="✅ 물리 모델 해석 완료!", state="complete")
        
    except ValueError as ve:
        status_box.update(label="🚨 해석 중단", state="error")
        st.error(str(ve)); st.stop()
    except Exception as e:
        status_box.update(label="🚨 해석 중단", state="error")
        st.error(f"예상치 못한 오류: {e}"); st.stop()

    df_res = pd.DataFrame(results)
    if global_audit_tracker:
        st.info("⚠️ **[계산 상태 알림]** 특정 구간에서 혼합물의 일부 물성치를 CoolProp이 계산하지 못하여 다음 가정(Fallback)이 적용되었습니다:\n" + "\n".join([f"- {m}" for m in global_audit_tracker]))

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

    st.subheader("📈 압력 및 온도 프로필 (PH-Flash 추적)")
    fig_prof = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['P (bar)'], name="Pressure", line=dict(color='red', width=3)), secondary_y=False)
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['T (°C)'], name="Temperature", line=dict(color='blue', dash='dash', width=2)), secondary_y=True)
    fig_prof.update_layout(height=400); fig_prof.update_yaxes(title_text="Pressure (bar)", secondary_y=False); fig_prof.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.subheader("📊 시뮬레이션 결과 데이터")
    st.dataframe(df_res, use_container_width=True)
