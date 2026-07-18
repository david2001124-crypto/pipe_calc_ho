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
st.set_page_config(page_title="Sequential Pipeline Simulator V4", page_icon="⚓", layout="wide")

@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def churchill_friction_factor(Re, e_D):
    """
    Churchill (1977) friction factor equation.
    Covers laminar, transition, and fully turbulent regimes.
    """
    if Re < 1e-10: return 0.0
    A = (2.457 * math.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * e_D)))**16
    B = (37530.0 / Re)**16
    f = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1/12.0)
    return f

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D, theta_deg, roughness, P_Pa):
    """
    Beggs and Brill (1973) multiphase flow calculation logic.
    """
    g = 9.81
    theta_rad = math.radians(theta_deg)
    
    v_m = v_SL + v_SG
    if v_m < 1e-6:
        return {"dP_dl_fric": 0, "dP_dl_elev": 0, "E_k": 0, "flow_regime": "Static", "f_tp": 0, "H_L": 1.0, "lambda_L": 1.0}
        
    lambda_L = v_SL / v_m
    lambda_L = max(min(lambda_L, 0.999), 0.001)
    
    N_Fr = (v_m**2) / (g * D)
    
    L1 = 31.6 * lambda_L**0.302
    L2 = 0.0009252 * lambda_L**-2.468
    L3 = 0.10 * lambda_L**-1.4516
    L4 = 0.5 * lambda_L**-6.738
    
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

    H_L0 = (a * lambda_L**b) / (N_Fr**c)
    H_L0 = max(min(H_L0, 0.999), lambda_L)

    C_corr = 0
    if theta_deg > 0: 
        if regime == "Segregated":
            C_corr = (1 - lambda_L) * math.log(lambda_L**2 * N_Fr * v_SL)
        elif regime == "Intermittent":
            C_corr = (1 - lambda_L) * math.log(lambda_L**0.1 * N_Fr * v_SL)
    
    beta = 1.0 + C_corr * (math.sin(1.8 * theta_rad) - (1/3)*math.sin(1.8 * theta_rad)**3)
    H_L = H_L0 * beta
    H_L = max(min(H_L, 0.999), lambda_L)

    rho_s = H_L * rho_L + (1 - H_L) * rho_G 
    rho_n = lambda_L * rho_L + (1 - lambda_L) * rho_G 
    
    mu_n = lambda_L * mu_L + (1 - lambda_L) * mu_G
    Re_n = (rho_n * v_m * D) / mu_n if mu_n > 0 else 1e6
    f_n = churchill_friction_factor(Re_n, roughness/D)
    
    y = max(lambda_L / (H_L**2), 1e-5)
    if 1.0 < y < 1.2:
        S = math.log(2.2 * y - 1.2)
    else:
        S = math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4)
    
    f_tp = f_n * math.exp(S)
    
    dP_dl_elev = rho_s * g * math.sin(theta_rad)
    dP_dl_fric = (f_tp * rho_n * v_m**2) / (2 * D)
    E_k = (rho_s * v_m * v_SG) / P_Pa
    E_k = min(E_k, 0.9) 

    return {
        "dP_dl_fric": dP_dl_fric, "dP_dl_elev": dP_dl_elev, "E_k": E_k,
        "flow_regime": regime, "H_L": H_L, "lambda_L": lambda_L,
        "v_m": v_m, "N_Fr": N_Fr, "Re_n": Re_n,
        "rho_n": rho_n, "exp_S": math.exp(S) 
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

st.title("⚓ 상선/해양 플랜트 순차적 다상유동 시뮬레이터 (V4.0)")
st.markdown("배관과 피팅을 블록처럼 조립하고, 자동 수렴(Auto-Mesh) 로직과 2D 시각화가 결합된 프로페셔널 버전입니다.")

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

st.header("2. 배관 공통 속성 및 외부 환경")
c1, c2, c3 = st.columns(3)
D_inner = c1.number_input("배관 내부 직경 (m)", value=None, format="%.4f")
thickness = c2.number_input("배관 두께 (m)", value=None, format="%.4f")
selected_sys_material = c3.selectbox("배관 재질 시스템 매핑 (ASME 기준)", [None] + list(material_db.MATERIAL_MAP.keys()))

c4, c5, c6, c7 = st.columns(4)
T_amb_C = c4.number_input("외부 환경 온도 (°C)", value=None, format="%.4f")
h_ext = c5.number_input("외부 대류 열전달 계수 (W/m²K)", value=None, format="%.4f")
t_ins = c6.number_input("보온재 두께 (m) [없으면 0]", value=None, format="%.4f")
k_ins = c7.number_input("보온재 열전도도 (W/mK)", value=None, format="%.4f")

if selected_sys_material:
    mat_info = material_db.MATERIAL_MAP[selected_sys_material]
    roughness = mat_info["roughness_m"]
    roughness_sci = f"{roughness:.2e}".replace("e", " x 10^")
    st.info(f"선택됨: **{mat_info['asme_category']} ({mat_info['asme_grade']})** | 절대 조도: **{roughness_sci} m**")

st.header("3. 순차적 파이프라인 빌더 (Sequential Builder)")
st.markdown("유동 방향에 따라 파이프(직관)와 피팅/밸브를 순서대로 추가하세요.")

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

with st.form("add_component_form"):
    comp_type = st.radio("추가할 요소 선택", ["Pipe Segment", "Fitting / Valve"], horizontal=True)
    
    ac1, ac2, ac3 = st.columns(3)
    # Pipe inputs
    p_len = ac1.number_input("직관 길이 (m)", min_value=0.0, value=10.0, format="%.4f")
    p_elev_type = ac2.selectbox("경사 입력", ["Angle (deg)", "Height (m)"])
    p_elev_val = ac3.number_input("경사값", value=0.0, format="%.4f")
    
    # Fitting inputs
    f_type = ac1.selectbox("피팅/밸브 종류", list(material_db.FITTING_LE_D_DB.keys()))
    f_qty = ac2.number_input("수량", min_value=1, value=1, step=1)
    
    submitted = st.form_submit_button("➕ 컴포넌트 추가")
    if submitted:
        if comp_type == "Pipe Segment":
            st.session_state.pipeline.append({
                "type": "Pipe", "length": p_len, "elev_type": p_elev_type, "elev_val": p_elev_val
            })
        else:
            st.session_state.pipeline.append({
                "type": "Fitting", "name": f_type, "qty": f_qty
            })
        st.success(f"{comp_type} 추가 완료!")

if st.button("🗑️ 전체 파이프라인 초기화"):
    st.session_state.pipeline = []
    st.rerun()

if st.session_state.pipeline:
    df_pipeline = pd.DataFrame(st.session_state.pipeline)
    st.table(df_pipeline)

st.divider()

if st.button("🚀 자동 수렴 2상 유동 시뮬레이션 시작", type="primary", use_container_width=True):
    # Validation Check
    missing_fields = []
    if not selected_fluids: missing_fields.append("유체 성분 선택")
    if any(v is None for v in fractions.values()): missing_fields.append("조성 몰 비율")
    core_inputs = {"입구 온도": T_inlet_C, "입구 압력": P_inlet_bar, "질량 유량": mass_flow,
                   "배관 내경": D_inner, "두께": thickness, "선박 재질": selected_sys_material, 
                   "외부 온도": T_amb_C, "대류 계수": h_ext, "보온재 두께": t_ins}
    for k, v in core_inputs.items():
        if v is None: missing_fields.append(k)
    if t_ins is not None and t_ins > 0 and k_ins is None:
        missing_fields.append("보온재 열전도율(k_ins)")
    if len(st.session_state.pipeline) == 0:
        missing_fields.append("파이프라인 컴포넌트 (최소 1개 이상 추가)")
        
    if missing_fields:
        st.error(f"🚨 입력 누락 감지! 다음 항목을 입력하세요: {', '.join(missing_fields)}")
        st.stop()

    total_frac = sum(fractions.values())
    norm_fractions = {k: v / total_frac for k, v in fractions.items()}
    fluid_string = list(norm_fractions.keys())[0] if len(norm_fractions) == 1 else "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])

    mat_info = material_db.MATERIAL_MAP[selected_sys_material]
    roughness = mat_info["roughness_m"]
    asme_table = material_db.RAW_DB[mat_info["asme_category"]][mat_info["asme_grade"]]
    
    D_outer = D_inner + 2 * thickness
    t_ins_calc, k_ins_calc = t_ins, (k_ins if k_ins is not None else 0.0)
    
    A_cross = math.pi * (D_inner / 2)**2
    G_flux = mass_flow / A_cross 

    def run_simulation(N_per_pipe):
        results = []
        T_current_K = T_inlet_C + 273.15
        P_current_Pa = P_inlet_bar * 100000
        L_cum = 0.0
        Z_cum = 0.0  # Elevation
        
        # 시작 지점 기록
        results.append({
            "Component": "Inlet", "Node": 0, "L_cum (m)": L_cum, "Z_cum (m)": Z_cum,
            "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15,
            "Phase": "-", "dP (Pa)": 0, "Regime": "Inlet"
        })

        for comp_idx, comp in enumerate(st.session_state.pipeline):
            if comp["type"] == "Pipe":
                length = comp["length"]
                dL = length / N_per_pipe
                
                # 경사 계산
                if comp["elev_type"] == "Height (m)":
                    h = comp["elev_val"]
                    if length == 0: angle_deg = 0
                    else: angle_deg = math.degrees(math.asin(max(min(h / length, 1.0), -1.0)))
                else:
                    angle_deg = comp["elev_val"]
                    
                dZ = dL * math.sin(math.radians(angle_deg))

                for i in range(N_per_pipe):
                    T_current_C = T_current_K - 273.15
                    k_pipe_current = np.interp(T_current_C, asme_table["T_C"], asme_table["k_W_mK"])
                    
                    try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                    except: phase_raw = "unknown"
                        
                    Q = -1.0 
                    is_twophase = False
                    if phase_raw == 'twophase':
                        is_twophase = True
                        try: Q = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        except: is_twophase = False 
                    
                    # 1-Phase
                    if not is_twophase:
                        rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        try: mu = PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        except: mu = 1e-5
                        Cp = PropsSI('C', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        k_fluid = PropsSI('L', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        
                        velocity = mass_flow / (rho * A_cross)
                        Re = (rho * velocity * D_inner) / mu
                        f_factor = churchill_friction_factor(Re, roughness/D_inner)
                        
                        dP_fric_node = ((f_factor * rho * velocity**2) / (2 * D_inner)) * dL
                        dP_elev_node = (rho * 9.81 * math.sin(math.radians(angle_deg))) * dL
                        dP_node_total = dP_fric_node + dP_elev_node
                        flow_regime = "1-Phase"
                        Pr = (Cp * mu) / k_fluid

                    # 2-Phase
                    else:
                        rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                        rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                        try: mu_L = PropsSI('V', 'P', P_current_Pa, 'Q', 0, fluid_string)
                        except: mu_L = 1e-3 
                        try: mu_G = PropsSI('V', 'P', P_current_Pa, 'Q', 1, fluid_string)
                        except: mu_G = 1e-5 
                        Cp = PropsSI('C', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        try: k_fluid = PropsSI('L', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                        except: k_fluid = 0.5
                        
                        v_SG = (mass_flow * Q) / (rho_G * A_cross)
                        v_SL = (mass_flow * (1 - Q)) / (rho_L * A_cross)
                        
                        bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, angle_deg, roughness, P_current_Pa)
                        
                        dP_fric_node = ((bb_result["f_tp"] * bb_result["rho_n"] * bb_result["v_m"]**2) / (2 * D_inner)) * dL
                        dP_elev_node = bb_result["dP_dl_elev"] * dL
                        dP_node_total = (dP_fric_node + dP_elev_node) / (1 - bb_result["E_k"])
                        
                        flow_regime = bb_result["flow_regime"]
                        Re = bb_result.get("Re_n", 1e5)
                        mu_mix = bb_result.get("lambda_L", 0.5) * mu_L + (1 - bb_result.get("lambda_L", 0.5)) * mu_G
                        Pr = (Cp * mu_mix) / k_fluid if k_fluid > 0 else 1.0

                    U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe_current, k_ins_calc, t_ins_calc, h_ext)
                    dT_dl = (U * math.pi * D_outer * ((T_amb_C + 273.15) - T_current_K)) / (mass_flow * Cp)
                    
                    P_current_Pa -= dP_node_total
                    T_current_K += dT_dl * dL
                    L_cum += dL
                    Z_cum += dZ
                    
                    results.append({
                        "Component": f"Pipe_{comp_idx+1}", "Node": i+1, 
                        "L_cum (m)": L_cum, "Z_cum (m)": Z_cum,
                        "P (bar)": P_current_Pa / 1e5, "T (°C)": T_current_K - 273.15,
                        "Phase": "2-Phase" if Q >= 0 else "1-Phase",
                        "dP (Pa)": dP_node_total, "Regime": flow_regime
                    })

            elif comp["type"] == "Fitting":
                f_name = comp["name"]
                qty = comp["qty"]
                L_e_total = (material_db.FITTING_LE_D_DB[f_name] * D_inner) * qty
                
                try: phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
                except: phase_raw = "unknown"
                Q = -1.0
                if phase_raw == 'twophase':
                    try: Q = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    except: pass
                
                if Q < 0:
                    rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    try: mu = PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                    except: mu = 1e-5
                    velocity = mass_flow / (rho * A_cross)
                    Re = (rho * velocity * D_inner) / mu
                    f_factor = churchill_friction_factor(Re, roughness/D_inner)
                    # 고저차 없이 마찰 길이(L_e)만 적용
                    dP_node_total = ((f_factor * rho * velocity**2) / (2 * D_inner)) * L_e_total
                    flow_regime = "1-Phase (Fitting)"
                else:
                    rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    try: mu_L = PropsSI('V', 'P', P_current_Pa, 'Q', 0, fluid_string)
                    except: mu_L = 1e-3 
                    try: mu_G = PropsSI('V', 'P', P_current_Pa, 'Q', 1, fluid_string)
                    except: mu_G = 1e-5 
                    v_SG = (mass_flow * Q) / (rho_G * A_cross)
                    v_SL = (mass_flow * (1 - Q)) / (rho_L * A_cross)
                    
                    # 밸브는 자체 각도가 없으므로 0도 처리
                    bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, 0.0, roughness, P_current_Pa)
                    
                    dP_fric_node = ((bb_result["f_tp"] * bb_result["rho_n"] * bb_result["v_m"]**2) / (2 * D_inner)) * L_e_total
                    dP_node_total = dP_fric_node / (1 - bb_result["E_k"])
                    flow_regime = bb_result["flow_regime"] + " (Fitting)"

                P_current_Pa -= dP_node_total
                # 피팅 통과 시 길이나 고도, 온도는 변하지 않음 (단열 마찰 가정)
                
                results.append({
                    "Component": f"Valve_{comp_idx+1} ({f_name})", "Node": "-", 
                    "L_cum (m)": L_cum, "Z_cum (m)": Z_cum,
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
            df_res, final_P, final_T = run_simulation(N_nodes)
            
            if prev_P is not None:
                dP_diff = abs((final_P - prev_P) / 1e5) # bar
                dT_diff = abs(final_T - prev_T) # K or C
                if dP_diff < 0.01 and dT_diff < 0.1:
                    converged = True
                    break
            
            prev_P, prev_T = final_P, final_T
            N_nodes *= 2 # Mesh 2배 세분화
            
        if converged: st.success(f"✅ 격자 독립성 달성! (반복 횟수: {iteration}, 구간당 Node: {N_nodes//2})")
        else: st.warning(f"⚠️ 최대 반복 도달 ({max_iter}회). 결과가 근사치일 수 있습니다. (Node: {N_nodes//2})")

    final_P_bar = df_res.iloc[-1]['P (bar)']
    final_T_C = df_res.iloc[-1]['T (°C)']
    
    m1, m2, m3 = st.columns(3)
    m1.metric("총 압력 강하 (Total dP)", f"{P_inlet_bar - final_P_bar:.4f} bar")
    m2.metric("출구 최종 압력 (Exit Pressure)", f"{final_P_bar:.4f} bar")
    m3.metric("출구 최종 온도 (Exit Temp)", f"{final_T_C:.4f} °C")

    # Plotly 2D Pipeline Sketch
    st.subheader("🗺️ 2D 파이프라인 스케치 (물리적 형태)")
    fig_2d = go.Figure()
    
    pipe_mask = df_res['Component'].str.startswith('Pipe') | (df_res['Component'] == 'Inlet')
    fig_2d.add_trace(go.Scatter(x=df_res[pipe_mask]['L_cum (m)'], y=df_res[pipe_mask]['Z_cum (m)'],
                                mode='lines', name='Pipeline', line=dict(color='blue', width=4)))
    
    valve_mask = df_res['Component'].str.startswith('Valve')
    if valve_mask.any():
        fig_2d.add_trace(go.Scatter(x=df_res[valve_mask]['L_cum (m)'], y=df_res[valve_mask]['Z_cum (m)'],
                                    mode='markers+text', name='Fittings/Valves',
                                    marker=dict(symbol='diamond', size=12, color='red'),
                                    text=df_res[valve_mask]['Component'].apply(lambda x: x.split('(')[0]),
                                    textposition="top center"))
        
    fig_2d.update_layout(xaxis_title="누적 길이 - L (m)", yaxis_title="누적 높이 - Z (m)",
                         height=400, showlegend=True, title="Flow Direction: Left ➔ Right")
    st.plotly_chart(fig_2d, use_container_width=True)

    # Plotly Property Profiles
    st.subheader("📈 압력 및 온도 프로필 (피팅 수직 강하 반영)")
    fig_prof = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['P (bar)'], name="Pressure (bar)", line=dict(color='red')), secondary_y=False)
    fig_prof.add_trace(go.Scatter(x=df_res['L_cum (m)'], y=df_res['T (°C)'], name="Temperature (°C)", line=dict(color='blue', dash='dash')), secondary_y=True)
    
    fig_prof.update_layout(height=400)
    fig_prof.update_yaxes(title_text="Pressure (bar)", secondary_y=False)
    fig_prof.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.subheader("📊 상세 노드 데이터표 (Pipe vs Fitting 분리)")
    st.dataframe(df_res, use_container_width=True)
