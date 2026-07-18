import streamlit as st
import pandas as pd
import numpy as np
import math
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import material_heat_transfer_coefficient_DB
# ---------------------------------------------------------
# UI & Configuration Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Ship Pipeline Simulator", page_icon="⚓", layout="wide")

@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

# ---------------------------------------------------------
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

# ---------------------------------------------------------
# ---------------------------------------------------------
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
        "v_m": v_m, "N_Fr": N_Fr, "Re_n": Re_n
    }

# ---------------------------------------------------------
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# ---------------------------------------------------------
st.title("⚓ 상선/해양 플랜트 다상유동 배관 시뮬레이터 (V3.1)")
st.markdown("ASME BPVC Section II Part D 열전도도 동적 DB 및 Engineering ToolBox 조도 데이터 연동")

st.header("1. 유체 성분 선택 (Fluid Composition)")
selected_fluids = st.multiselect("CoolProp 지원 전유체 중 선택", AVAILABLE_FLUIDS, default=None)
fractions = {}
if selected_fluids:
    cols = st.columns(len(selected_fluids))
    for i, fluid in enumerate(selected_fluids):
        val = cols[i].number_input(f"{fluid} 몰 비율", value=None, step=None, format="%.4f")
        fractions[fluid] = val

st.header("2. 운전 조건 입력 (Operating Conditions)")
col1, col2, col3 = st.columns(3)
T_inlet_C = col1.number_input("입구 온도 (°C)", value=None, step=None, format="%.4f")
P_inlet_bar = col2.number_input("입구 압력 (bar)", value=None, step=None, format="%.4f")
mass_flow = col3.number_input("질량 유량 (kg/s)", value=None, step=None, format="%.4f")

st.header("3. 배관 기하학 & 선박 계통 재질 매핑")
col_p1, col_p2, col_p3 = st.columns(3)
L_total = col_p1.number_input("배관 총 길이 (m)", value=None, step=None, format="%.4f")
elev_type = col_p2.selectbox("경사 입력 방식", ["Angle (deg)", "Height (m)"])
elev_val = col_p3.number_input(f"경사값 ({elev_type})", value=None, step=None, format="%.4f")

col_p4, col_p5 = st.columns(2)
D_inner = col_p4.number_input("배관 내부 직경 (m)", value=None, step=None, format="%.4f")
thickness = col_p5.number_input("배관 두께 (m)", value=None, step=None, format="%.4f")

material_keys = list(material_db.SHIPBUILDING_MAP.keys())
selected_sys_material = st.selectbox("⚓ 선박 시스템 및 배관 강재 선택", [None] + material_keys)
if selected_sys_material:
    mat_info = material_db.SHIPBUILDING_MAP[selected_sys_material]
    st.info(f"선택 반영: 절대 조도 **{mat_info['roughness_m']} m** | ASME 매핑: **{mat_info['asme_category']} - {mat_info['asme_grade']}**")

col_ins1, col_ins2 = st.columns(2)
t_ins = col_ins1.number_input("보온재 두께 (m, 없으면 0)", value=None, step=None, format="%.4f")
k_ins = col_ins2.number_input("보온재 열전도율 (W/m·K)", value=None, step=None, format="%.4f")

st.header("4. 주변 환경 & 5. 수치해석 노드 설정")
col_env1, col_env2, col_env3 = st.columns(3)
T_amb_C = col_env1.number_input("외부 환경 온도 (°C)", value=None, step=None, format="%.4f")
h_ext = col_env2.number_input("외부 대류 계수 (W/m²·K)", value=None, step=None, format="%.4f")
increments = col_env3.number_input("배관 분할 개수 (Nodes)", value=None, step=None, format="%.0f")

st.divider()

# ---------------------------------------------------------
# ---------------------------------------------------------
if st.button("🚀 압력 강하 시뮬레이션 시작", type="primary", use_container_width=True):
    
    # 1. Validation Check
    missing_fields = []
    if not selected_fluids: missing_fields.append("유체 성분 선택")
    if any(v is None for v in fractions.values()): missing_fields.append("조성 몰 비율")
    core_inputs = {"입구 온도": T_inlet_C, "입구 압력": P_inlet_bar, "질량 유량": mass_flow,
                   "총 배관 길이": L_total, "경사값": elev_val, "배관 내경": D_inner, "두께": thickness,
                   "선박 재질": selected_sys_material, "외부 온도": T_amb_C, "외부 대류 계수": h_ext, "분할 개수": increments, "보온재 두께": t_ins}
    
    for k, v in core_inputs.items():
        if v is None: missing_fields.append(k)
    if t_ins is not None and t_ins > 0 and k_ins is None:
        missing_fields.append("보온재 열전도율(k_ins)")
        
    if missing_fields:
        st.error(f"🚨 입력 누락 감지! 다음 항목을 입력하세요: {', '.join(missing_fields)}")
        st.stop()

    # 2. Setup Logic
    total_frac = sum(fractions.values())
    if total_frac <= 0: st.error("몰 비율 합은 0보다 커야 합니다."); st.stop()
    norm_fractions = {k: v / total_frac for k, v in fractions.items()}
    fluid_string = list(norm_fractions.keys())[0] if len(norm_fractions) == 1 else "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])

    if elev_type == "Height (m)":
        if L_total == 0 or abs(elev_val) > L_total: st.error("경사 높이는 총 길이를 초과할 수 없습니다."); st.stop()
        elevation_angle = math.degrees(math.asin(elev_val / L_total))
    else:
        elevation_angle = elev_val

    mat_info = material_db.SHIPBUILDING_MAP[selected_sys_material]
    roughness = mat_info["roughness_m"]
    asme_table = material_db.RAW_DB[mat_info["asme_category"]][mat_info["asme_grade"]]
    
    dL = L_total / increments
    T_current_K = T_inlet_C + 273.15
    P_current_Pa = P_inlet_bar * 100000
    D_outer = D_inner + 2 * thickness
    t_ins_calc, k_ins_calc = t_ins, (k_ins if k_ins is not None else 0.0)
    
    results = []
    first_node_debug = {}
    progress_bar = st.progress(0)
    
    try:
        for i in range(int(increments)):
            T_current_C = T_current_K - 273.15
            
            # Dynamic Thermal Conductivity Interpolation (ASME Part D Table)
            k_pipe_current = np.interp(T_current_C, asme_table["T_C"], asme_table["k_W_mK"])
            
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
                try: mu = PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                except ValueError:
                    mu_sum = sum(frac * PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, f_name) for f_name, frac in norm_fractions.items())
                    mu = mu_sum if mu_sum > 0 else 1e-5
                
                Cp = PropsSI('C', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                k_fluid = PropsSI('L', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                
                velocity = mass_flow / (rho * math.pi * (D_inner/2)**2)
                Re = (rho * velocity * D_inner) / mu
                f_factor = churchill_friction_factor(Re, roughness/D_inner)
                
                dP_dl_fric = (f_factor * rho * velocity**2) / (2 * D_inner)
                dP_dl_elev = rho * 9.81 * math.sin(math.radians(elevation_angle))
                dP_dl_total = dP_dl_fric + dP_dl_elev
                flow_regime = "1-Phase"
                Pr = (Cp * mu) / k_fluid
                
                if i == 0:
                    first_node_debug = {"상태": "단상 (1-Phase)", "밀도": rho, "점도": mu, "Re": Re, "k_pipe": k_pipe_current}
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
                
                v_SG = (mass_flow * Q) / (rho_G * math.pi * (D_inner/2)**2)
                v_SL = (mass_flow * (1 - Q)) / (rho_L * math.pi * (D_inner/2)**2)
                
                bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, elevation_angle, roughness, P_current_Pa)
                dP_dl_total = (bb_result["dP_dl_fric"] + bb_result["dP_dl_elev"]) / (1 - bb_result["E_k"])
                flow_regime = bb_result["flow_regime"]
                
                Re = bb_result.get("Re_n", 1e5)
                mu_mix = bb_result.get("lambda_L", 0.5) * mu_L + (1 - bb_result.get("lambda_L", 0.5)) * mu_G
                Pr = (Cp * mu_mix) / k_fluid if k_fluid > 0 else 1.0
                
                if i == 0:
                    first_node_debug = {"상태": "2상 (2-Phase)", "Quality": Q, "Regime": flow_regime, "H_L": bb_result['H_L'], "k_pipe": k_pipe_current}

            U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe_current, k_ins_calc, t_ins_calc, h_ext)
            dT_dl = (U * math.pi * D_outer * ((T_amb_C + 273.15) - T_current_K)) / (mass_flow * Cp)
            
            results.append({
                "Node": i, "길이 (m)": i * dL, "압력 (bar)": P_current_Pa / 100000,
                "온도 (°C)": T_current_C, "Phase": "2-Phase" if Q >= 0 else "1-Phase",
                "Quality": Q if Q >= 0 else 0, "Regime": flow_regime, "dP/dL (Pa/m)": dP_dl_total, "k_pipe": k_pipe_current
            })
            
            P_current_Pa -= dP_dl_total * dL
            T_current_K += dT_dl * dL
            
            if P_current_Pa <= 10000:
                st.warning("🚨 압력 고갈 (대기압 이하)로 계산 조기 종료"); break
            progress_bar.progress((i + 1) / int(increments))
            
        progress_bar.empty()
        
        df_res = pd.DataFrame(results)
        final_P, final_T = df_res.iloc[-1]['압력 (bar)'], df_res.iloc[-1]['온도 (°C)']
        
        st.success("✅ 시뮬레이션 계산 완료!")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 압력 강하 (Total dP)", f"{P_inlet_bar - final_P:.4f} bar")
        m2.metric("출구 최종 압력 (Exit Pressure)", f"{final_P:.4f} bar")
        m3.metric("출구 최종 온도 (Exit Temp)", f"{final_T:.4f} °C")
        
        with st.expander("🔍 상세 투명 계산 과정 (Glass-box Node-0)"):
            st.markdown("ASME Table TCD 동적 열전도도 매핑 및 Beggs & Brill 적용 확인")
            st.json(first_node_debug)

        c1, c2 = st.columns(2)
        c1.line_chart(df_res.set_index('길이 (m)')['압력 (bar)'], color="#ff4b4b")
        c2.line_chart(df_res.set_index('길이 (m)')['온도 (°C)'], color="#0068c9")
        st.dataframe(df_res, use_container_width=True)

    except Exception as e:
        st.error(f"🚨 열역학 계산 오류 발생: {e}")
