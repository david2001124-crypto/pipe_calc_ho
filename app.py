import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import material_db

# ---------------------------------------------------------
# UI & Configuration Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Sequential Pipeline Simulator V4.3", page_icon="⚓", layout="wide")

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
    ASME 재질 DB의 온도 범위를 벗어날 경우, 선형 외삽법(Linear Extrapolation)을 
    적용하여 물리적으로 타당한 열전도도(k) 값을 추정하는 함수
    """
    T_arr = np.array(T_list)
    k_arr = np.array(k_list)
    
    if len(T_arr) < 2: 
        return k_arr[0]
        
    if T_target < T_arr[0]:
        # 하한값 이탈 시 선형 외삽 (하위 2개 점 이용)
        slope = (k_arr[1] - k_arr[0]) / (T_arr[1] - T_arr[0])
        return k_arr[0] + slope * (T_target - T_arr[0])
    elif T_target > T_arr[-1]:
        # 상한값 이탈 시 선형 외삽 (상위 2개 점 이용)
        slope = (k_arr[-1] - k_arr[-2]) / (T_arr[-1] - T_arr[-2])
        return k_arr[-1] + slope * (T_target - T_arr[-1])
    else:
        # 범위 내에 있으면 기본 선형 보간
        return float(np.interp(T_target, T_arr, k_arr))

def get_robust_prop(prop_char, T, P, fluid_str, norm_fractions, fallback_val, tracker_set):
    """
    CoolProp에서 혼합물 점도('V')나 열전도도('L') 계산 실패 시,
    각 순수 성분의 몰 분율로 가중 평균(Mixing Rule)을 적용하고,
    적용된 가정을 Audit Trail(추적기)에 기록하는 고신뢰성 헬퍼 함수
    """
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
                
        tracker_set.add(f"{prop_name} 계산 불가 ➔ 기본 상수({fallback_val}) 강제 적용")
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
        return {"dP_dl_fric": 0, "dP_dl_elev": 0, "E_k": 0, "flow_regime": "Static", "f_tp": 0, "H_L": 1.0, "lambda_L": 1.0, "rho_n": rho_L}
        
    lambda_L = v_SL / v_m
    lambda_L = max(min(lambda_L, 0.999), 0.001)
    
    N_Fr = (v_m**2) / (g * D)
    L1, L2 = 31.6 * lambda_L**0.302, 0.0009252 * lambda_L**-2.468
    L3, L4 = 0.10 * lambda_L**-1.4516, 0.5 * lambda_L**-6.738
    
    regime = "Unknown"
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2):
        regime = "Segregated"
        a, b, c = 0.98, 0.4846, 0.0868
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4):
        regime = "Intermittent"
        a, b, c = 0.845, 0.5351, 0.0173
    else:
        regime = "Distributed"
        a, b, c = 1.065, 0.5824, 0.0609

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
    if 1.0 < y < 1.2: S = math.log(2.2 * y - 1.2)
    else: S = math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4)
    
    f_tp = f_n * math.exp(S)
    dP_dl_elev = rho_s * g * math.sin(theta_rad)
    E_k = min((rho_s * v_m * v_SG) / P_Pa, 0.9) 

    return {
        "dP_dl_fric": (f_tp * rho_n * v_m**2) / (2 * D), "dP_dl_elev": dP_dl_elev, "E_k": E_k,
        "flow_regime": regime, "H_L": H_L, "lambda_L": lambda_L,
        "v_m": v_m, "N_Fr": N_Fr, "Re_n": Re_n, "rho_n": rho_n, "exp_S": math.exp(S), "f_tp": f_tp 
    }

def calculate_heat_transfer(Re, Pr, k_fluid, D_i, D_o, k_pipe, k_ins, t_ins, h_o):
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36
    h_i = (Nu * k_fluid) / D_i if D_i > 0 else 1e-5
    
    R_i = D_o / (D_i * h_i)
    R_pipe = (D_o / (2 * k_pipe)) * math.log(D_o / D_i) if k_pipe > 0 else 0
    R_ins = 0
    if t_ins > 0 and k_ins > 0:
        D_ins_o = D_o + 2 * t_ins
        R_ins = (D_o / (2 * k_ins)) * math.log(D_ins_o / D_o)
    R_o = 1.0 / h_o if h_o > 0 else 0
    return 1.0 / (R_i + R_pipe + R_ins + R_o)

st.title("⚓ 상선/해양 플랜트 다상유동 시뮬레이터 (V4.3)")
st.markdown("**(선형 외삽법 적용 및 자동 격자수렴 AI 엔진 탑재)**")

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
        val = f_cols[i].number_input(f"{fluid} 몰 비율", value=None, format="%.4f")
        fractions[fluid] = val

st.header("2. 순차적 파이프라인 빌더 (Sequential Builder)")
st.markdown("파이프 구간과 피팅/밸브를 원하는 순서대로 조립하세요. 각 배관의 속성(재질, 직경)을 독립적으로 설정할 수 있습니다.")

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

with st.form("add_component_form"):
    comp_type = st.radio("추가할 컴포넌트", ["Pipe Segment (배관)", "Fitting / Valve (밸브 및 피팅)"], horizontal=True)
    
    st.markdown("##### 세부 속성 입력")
    ac1, ac2, ac3 = st.columns(3)
    p_len = ac1.number_input("직관 길이 (m)", min_value=0.0, value=10.0, format="%.4f")
    p_elev_type = ac2.selectbox("경사 입력 방식", ["Angle (deg)", "Height (m)"])
    p_elev_val = ac3.number_input("경사/높이 값", value=0.0, format="%.4f")
    
    pc1, pc2, pc3 = st.columns(3)
    D_inner = pc1.number_input("내부 직경 (m)", value=0.1, format="%.4f")
    thickness = pc2.number_input("배관 두께 (m)", value=0.005, format="%.4f")
    selected_sys_material = pc3.selectbox("배관 재질 선택 (ASME)", list(material_db.MATERIAL_MAP.keys()))
    
    fc1, fc2 = st.columns(2)
    f_type = fc1.selectbox("피팅/밸브 종류", list(material_db.FITTING_LE_D_DB.keys()))
    f_qty = fc2.number_input("수량", min_value=1, value=1, step=1)
    
    submitted = st.form_submit_button("➕ 컴포넌트 추가")
    if submitted:
        if "Pipe" in comp_type:
            st.session_state.pipeline.append({
                "type": "Pipe", "length": p_len, "elev_type": p_elev_type, "elev_val": p_elev_val,
                "D_inner": D_inner, "thickness": thickness, "material": selected_sys_material
            })
        else:
            st.session_state.pipeline.append({
                "type": "Fitting", "name": f_type, "qty": f_qty
            })
        st.success(f"{comp_type} 추가 완료!")

if st.session_state.pipeline:
    st.markdown("##### 🧱 현재 구성된 파이프라인 목록 (Flow: 위 ➔ 아래)")
    for idx, comp in enumerate(st.session_state.pipeline):
        cc1, cc2, cc3, cc4 = st.columns([0.7, 0.1, 0.1, 0.1])
        if comp["type"] == "Pipe":
            desc = f"**[{idx+1}] 배관:** L={comp['length']}m | ID={comp['D_inner']}m | 재질: {comp['material'].split(' / ')[0]}"
        else:
            desc = f"**[{idx+1}] 밸브/피팅:** {comp['name']} (x{comp['qty']})"
            
        cc1.write(desc)
        if cc2.button("⬆️", key=f"up_{idx}") and idx > 0:
            st.session_state.pipeline[idx-1], st.session_state.pipeline[idx] = st.session_state.pipeline[idx], st.session_state.pipeline[idx-1]
            st.rerun()
        if cc3.button("⬇️", key=f"down_{idx}") and idx < len(st.session_state.pipeline)-1:
            st.session_state.pipeline[idx+1], st.session_state.pipeline[idx] = st.session_state.pipeline[idx], st.session_state.pipeline[idx+1]
            st.rerun()
        if cc4.button("🗑️", key=f"del_{idx}"):
            st.session_state.pipeline.pop(idx)
            st.rerun()

st.header("3. 외부 환경 (전체 공통 적용)")
ec1, ec2, ec3, ec4 = st.columns(4)
T_amb_C = ec1.number_input("외부 온도 (°C)", value=None, format="%.4f")
h_ext = ec2.number_input("외부 열전달 계수 (W/m²K)", value=None, format="%.4f")
t_ins = ec3.number_input("보온재 두께 (m)", value=0.0, format="%.4f")
k_ins = ec4.number_input("보온재 열전도도 (W/mK)", value=0.0, format="%.4f")

st.divider()

if st.button("🚀 자동 수렴 2상 유동 시뮬레이션 시작", type="primary", use_container_width=True):
    missing = []
    if not selected_fluids: missing.append("유체 성분")
    if any(v is None for v in fractions.values()): missing.append("몰 비율")
    if T_inlet_C is None: missing.append("입구 온도")
    if P_inlet_bar is None: missing.append("입구 압력")
    if mass_flow is None: missing.append("질량 유량")
    if T_amb_C is None: missing.append("외부 온도")
    if h_ext is None: missing.append("외부 열전달 계수")
    if len(st.session_state.pipeline) == 0: missing.append("파이프라인 컴포넌트")
        
    if missing:
        st.error(f"🚨 다음 필수 항목을 입력하세요: {', '.join(missing)}")
        st.stop()

    total_frac = sum(fractions.values())
    norm_fractions = {k: v / total_frac for k, v in fractions.items()}
    fluid_string = list(norm_fractions.keys())[0] if len(norm_fractions) == 1 else "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])

    global_audit_tracker = set()

    def run_simulation(N_per_pipe, audit_tracker):
        results = []
        T_current_K = T_inlet_C + 273.15
        P_current_Pa = P_inlet_bar * 100000
        L_cum = 0.0
        Z_cum = 0.0 
        
        # 이전 배관 정보를 저장할 초기 변수 (가장 처음 피팅이 올 경우를 대비)
        curr_D_inner, curr_thickness, curr_roughness, curr_mat_info = 0.1, 0.005, 4.5e-5, list(material_db.MATERIAL_MAP.values())[0]
        
        results.append({
            "Component": "Inlet", "Node": 0, "L_cum (m)": L_cum, "Z_cum (m)": Z_cum,
            "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15,
            "Phase": "-", "dP (Pa)": 0, "Regime": "Inlet Node"
        })

        for comp_idx, comp in enumerate(st.session_state.pipeline):
            if comp["type"] == "Pipe":
                # 파이프 컴포넌트: 실제 길이가 존재하며, Node(격자)를 나누어 계산
                curr_D_inner = comp["D_inner"]
                curr_thickness = comp["thickness"]
                curr_mat_info = material_db.MATERIAL_MAP[comp["material"]]
                curr_roughness = curr_mat_info["roughness_m"]
                asme_table = material_db.RAW_DB[curr_mat_info["asme_category"]][curr_mat_info["asme_grade"]]
                D_outer = curr_D_inner + 2 * curr_thickness
                A_cross = math.pi * (curr_D_inner / 2)**2
                
                length = comp["length"]
                dL = length / N_per_pipe if N_per_pipe > 0 else 0
                if comp["elev_type"] == "Height (m)":
                    angle_deg = 0 if length == 0 else math.degrees(math.asin(max(min(comp["elev_val"] / length, 1.0), -1.0)))
                else:
                    angle_deg = comp["elev_val"]
                dZ = dL * math.sin(math.radians(angle_deg))

                for i in range(N_per_pipe):
                    T_current_C = T_current_K - 273.15
                    
                    # V4.3 신규: 외삽(Extrapolation) 함수 적용
                    k_pipe_current = get_k_pipe_extrapolated(T_current_C, asme_table["T_C"], asme_table["k_W_mK"])
                    
                    try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                    except: phase_raw = "unknown"
                        
                    Q = -1.0 
                    is_twophase = False
                    if phase_raw == 'twophase':
                        is_twophase = True
                        try: Q = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        except: is_twophase = False 
                    
                    if not is_twophase:
                        rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        mu = get_robust_prop('V', T_current_K, P_current_Pa, fluid_string, norm_fractions, 1e-5, audit_tracker)
                        Cp = get_robust_prop('C', T_current_K, P_current_Pa, fluid_string, norm_fractions, 2000, audit_tracker)
                        k_fluid = get_robust_prop('L', T_current_K, P_current_Pa, fluid_string, norm_fractions, 0.1, audit_tracker)
                        
                        velocity = mass_flow / (rho * A_cross)
                        Re = (rho * velocity * curr_D_inner) / mu
                        f_factor = churchill_friction_factor(Re, curr_roughness/curr_D_inner)
                        
                        dP_fric_node = ((f_factor * rho * velocity**2) / (2 * curr_D_inner)) * dL
                        dP_elev_node = (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
                        dP_node_total = dP_fric_node + dP_elev_node
                        flow_regime = "1-Phase"
                        Pr = (Cp * mu) / k_fluid if k_fluid > 0 else 1.0

                    else:
                        rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                        rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                        mu_L = get_robust_prop('V', T_current_K, P_current_Pa, fluid_string, norm_fractions, 1e-3, audit_tracker)
                        mu_G = get_robust_prop('V', T_current_K, P_current_Pa, fluid_string, norm_fractions, 1e-5, audit_tracker)
                        Cp = get_robust_prop('C', T_current_K, P_current_Pa, fluid_string, norm_fractions, 2500, audit_tracker)
                        k_fluid = get_robust_prop('L', T_current_K, P_current_Pa, fluid_string, norm_fractions, 0.2, audit_tracker)
                        
                        v_SG = (mass_flow * Q) / (rho_G * A_cross)
                        v_SL = (mass_flow * (1 - Q)) / (rho_L * A_cross)
                        
                        bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, curr_D_inner, angle_deg, curr_roughness, P_current_Pa)
                        dP_fric_node = ((bb_result["f_tp"] * bb_result["rho_n"] * bb_result["v_m"]**2) / (2 * curr_D_inner)) * dL
                        dP_elev_node = bb_result["dP_dl_elev"] * dL
                        dP_node_total = (dP_fric_node + dP_elev_node) / (1 - bb_result["E_k"])
                        
                        flow_regime = bb_result["flow_regime"]
                        Re = bb_result.get("Re_n", 1e5)
                        mu_mix = bb_result.get("lambda_L", 0.5) * mu_L + (1 - bb_result.get("lambda_L", 0.5)) * mu_G
                        Pr = (Cp * mu_mix) / k_fluid if k_fluid > 0 else 1.0

                    U = calculate_heat_transfer(Re, Pr, k_fluid, curr_D_inner, D_outer, k_pipe_current, k_ins, t_ins, h_ext)
                    dT_dl = (U * math.pi * D_outer * ((T_amb_C + 273.15) - T_current_K)) / (mass_flow * Cp)
                    
                    P_current_Pa -= dP_node_total
                    T_current_K += dT_dl * dL
                    L_cum += dL
                    Z_cum += dZ
                    
                    results.append({
                        "Component": f"Pipe_{comp_idx+1}", "Node": i+1, 
                        "L_cum (m)": L_cum, "Z_cum (m)": Z_cum,
                        "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15,
                        "Phase": "2-Phase" if is_twophase else "1-Phase",
                        "dP (Pa)": dP_node_total, "Regime": flow_regime
                    })

            elif comp["type"] == "Fitting":
                # 피팅/밸브 컴포넌트: 물리적 길이는 늘어나지 않으나 등가 길이(Le)를 적용하여 압력 강하만 유발
                f_name = comp["name"]
                qty = comp["qty"]
                # 직전 파이프의 직경(curr_D_inner) 상속
                L_e_total = (material_db.FITTING_LE_D_DB[f_name] * curr_D_inner) * qty
                A_cross = math.pi * (curr_D_inner / 2)**2
                
                try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                except: phase_raw = "unknown"
                
                Q = -1.0
                if phase_raw == 'twophase':
                    try: Q = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    except: pass
                
                if Q < 0:
                    rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    mu = get_robust_prop('V', T_current_K, P_current_Pa, fluid_string, norm_fractions, 1e-5, audit_tracker)
                    velocity = mass_flow / (rho * A_cross)
                    Re = (rho * velocity * curr_D_inner) / mu
                    f_factor = churchill_friction_factor(Re, curr_roughness/curr_D_inner)
                    # L_e_total을 적용 (고도 변화 없음)
                    dP_node_total = ((f_factor * rho * velocity**2) / (2 * curr_D_inner)) * L_e_total
                    flow_regime = "Fitting (1-Phase)"
                else:
                    rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    mu_L = get_robust_prop('V', T_current_K, P_current_Pa, fluid_string, norm_fractions, 1e-3, audit_tracker)
                    mu_G = get_robust_prop('V', T_current_K, P_current_Pa, fluid_string, norm_fractions, 1e-5, audit_tracker)
                    v_SG = (mass_flow * Q) / (rho_G * A_cross)
                    v_SL = (mass_flow * (1 - Q)) / (rho_L * A_cross)
                    
                    # 피팅에서는 고도 변화각(theta)=0 으로 가정
                    bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, curr_D_inner, 0.0, curr_roughness, P_current_Pa)
                    # L_e_total을 가상의 직관 길이로 사용하여 2상 마찰 적용
                    dP_fric_node = ((bb_result["f_tp"] * bb_result["rho_n"] * bb_result["v_m"]**2) / (2 * curr_D_inner)) * L_e_total
                    dP_node_total = dP_fric_node / (1 - bb_result["E_k"])
                    flow_regime = bb_result["flow_regime"] + " (Fitting L_e)"

                P_current_Pa -= dP_node_total
                
                results.append({
                    "Component": f"Valve/Fit_{comp_idx+1} ({f_name})", "Node": "Point Drop", 
                    "L_cum (m)": L_cum, "Z_cum (m)": Z_cum, # 길이와 높이 증가는 없음
                    "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15,
                    "Phase": "2-Phase" if Q >= 0 else "1-Phase",
                    "dP (Pa)": dP_node_total, "Regime": flow_regime
                })

        return pd.DataFrame(results), P_current_Pa, T_current_K

    with st.spinner("🤖 인공지능 Auto-Mesh 수렴 계산 중..."):
        N_nodes = 5
        max_iter = 6
        prev_P, prev_T = None, None
        converged = False
        
        for iteration in range(1, max_iter + 1):
            global_audit_tracker.clear() # 반복마다 트래커 초기화
            df_res, final_P, final_T = run_simulation(N_nodes, global_audit_tracker)
            
            if prev_P is not None:
                dP_diff = abs((final_P - prev_P) / 1e5)
                dT_diff = abs(final_T - prev_T)
                if dP_diff < 0.01 and dT_diff < 0.1:
                    converged = True
                    break
            
            prev_P, prev_T = final_P, final_T
            N_nodes *= 2
            
        if converged: 
            st.success(f"✅ 격자 독립성 달성 (수렴 완료)! (반복: {iteration}회, 구간당 Node: {N_nodes//2})")
        else: 
            st.warning(f"⚠️ 최대 반복 도달 ({max_iter}회). 결과가 근사치일 수 있습니다. (Node: {N_nodes//2})")

    if global_audit_tracker:
        audit_msg = "⚠️ **[계산 상태 알림]** 특정 온도/압력 구간에서 혼합물의 일부 물성치를 데이터베이스에서 찾을 수 없어, 다음의 예외 규칙(Fallback)이 적용되었습니다:\n"
        for msg in global_audit_tracker:
            audit_msg += f"\n- {msg}"
        st.info(audit_msg)

    final_P_bar = df_res.iloc[-1]['P (bar)']
    final_T_C = df_res.iloc[-1]['T (°C)']
    
    m1, m2, m3 = st.columns(3)
    m1.metric("총 압력 강하 (Total dP)", f"{P_inlet_bar - final_P_bar:.4f} bar")
    m2.metric("최종 출구 압력 (Exit Pressure)", f"{final_P_bar:.4f} bar")
    m3.metric("최종 출구 온도 (Exit Temp)", f"{final_T_C:.4f} °C")

    # 2D 파이프라인 물리 스케치
    st.subheader("🗺️ 2D 파이프라인 물리 스케치")
    fig_2d = go.Figure()
    
    pipe_mask = df_res['Component'].str.startswith('Pipe') | (df_res['Component'] == 'Inlet')
    fig_2d.add_trace(go.Scatter(x=df_res[pipe_mask]['L_cum (m)'], y=df_res[pipe_mask]['Z_cum (m)'],
                                mode='lines', name='Pipeline Path', line=dict(color='blue', width=4)))
    
    valve_mask = df_res['Component'].str.startswith('Valve/Fit')
    if valve_mask.any():
        fig_2d.add_trace(go.Scatter(x=df_res[valve_mask]['L_cum (m)'], y=df_res[valve_mask]['Z_cum (m)'],
                                    mode='markers+text', name='Fittings / Valves',
                                    marker=dict(symbol='diamond', size=14, color='red', line=dict(width=2, color='darkred')),
                                    text=df_res[valve_mask]['Component'].apply(lambda x: x.split('(')[1].replace(')','')),
                                    textposition="top center"))
        
    fig_2d.update_layout(xaxis_title="누적 길이 - X (m)", yaxis_title="누적 고도 - Z (m)",
                         height=400, showlegend=True, title="유동 방향: Left ➔ Right")
    st.plotly_chart(fig_2d, use_container_width=True)

    # 압력 및 온도 프로필 (결과 그래프)
    st.subheader("📈 압력 및 온도 프로필")
    fig_prof = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['P (bar)'], name="Pressure (bar)", line=dict(color='red', width=3)), secondary_y=False)
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['T (°C)'], name="Temperature (°C)", line=dict(color='blue', dash='dash', width=2)), secondary_y=True)
    fig_prof.update_layout(height=400, title="배관 길이에 따른 유동 상태 (피팅/밸브 위치에서 압력 수직 강하 발생)")
    fig_prof.update_yaxes(title_text="Pressure (bar)", secondary_y=False)
    fig_prof.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig_prof, use_container_width=True)

    # 데이터 테이블 뷰
    st.subheader("📊 시뮬레이션 결과 상세 데이터")
    st.dataframe(df_res, use_container_width=True)
