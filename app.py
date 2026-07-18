import streamlit as st
import pandas as pd
import numpy as np
import math
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import time

st.set_page_config(page_title="Multiphase Pipeline Simulator", page_icon="🚰", layout="wide")

@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

def churchill_friction_factor(Re, e_D):
    """
    Churchill (1977) equation for Darcy friction factor.
    Valid for laminar, transitional, and turbulent flow.
    """
    if Re < 1e-10: return 0.0
    A = (2.457 * math.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * e_D)))**16
    B = (37530.0 / Re)**16
    f = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1/12.0)
    return f

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D, theta_deg, roughness, P_Pa):
    """
    Beggs and Brill (1973) 다상 유동 압력 강하 모델
    """
    g = 9.81
    theta_rad = math.radians(theta_deg)
    
    v_m = v_SL + v_SG
    if v_m < 1e-6:
        return {"dP_dl_fric": 0, "dP_dl_elev": 0, "E_k": 0, "flow_regime": "Static", "f_tp": 0, "H_L": 1.0, "lambda_L": 1.0}
        
    lambda_L = v_SL / v_m # No-slip holdup
    lambda_L = max(min(lambda_L, 0.999), 0.001) # 방어 로직
    N_Fr = (v_m**2) / (g * D) # Froude number
    
    # 1. Flow Regime Determination (Simplified Boundaries)
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

    # 2. Horizontal Holdup (H_L0)
    H_L0 = (a * lambda_L**b) / (N_Fr**c)
    H_L0 = max(min(H_L0, 0.999), lambda_L) 

    # 3. Inclination Correction Factor (beta)
    C_corr = 0
    if theta_deg > 0: # Uphill
        if regime == "Segregated": C_corr = (1 - lambda_L) * math.log(lambda_L**2 * N_Fr * v_SL) # Simplified
        elif regime == "Intermittent": C_corr = (1 - lambda_L) * math.log(lambda_L**0.1 * N_Fr * v_SL)
    
    beta = 1.0 + C_corr * (math.sin(1.8 * theta_rad) - (1/3)*math.sin(1.8 * theta_rad)**3)
    H_L = H_L0 * beta
    H_L = max(min(H_L, 0.999), lambda_L) # 실제 체류율

    # 4. Densities
    rho_s = H_L * rho_L + (1 - H_L) * rho_G # Slip density (Elev)
    rho_n = lambda_L * rho_L + (1 - lambda_L) * rho_G # No-slip density (Fric)
    
    # 5. Friction Factor
    mu_n = lambda_L * mu_L + (1 - lambda_L) * mu_G
    Re_n = (rho_n * v_m * D) / mu_n if mu_n > 0 else 1e6
    f_n = churchill_friction_factor(Re_n, roughness/D)
    
    # Friction Multiplier (e^S)
    y = max(lambda_L / (H_L**2), 1e-5)
    if 1.0 < y < 1.2:
        S = math.log(2.2 * y - 1.2)
    else:
        S = math.log(y) / (-0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y))**2 + 0.01853 * (math.log(y))**4)
    
    f_tp = f_n * math.exp(S)
    
    # 6. Pressure Drop Components
    dP_dl_elev = rho_s * g * math.sin(theta_rad)
    dP_dl_fric = (f_tp * rho_n * v_m**2) / (2 * D)
    E_k = (rho_s * v_m * v_SG) / P_Pa # Acceleration term
    E_k = min(E_k, 0.9) # 방어 로직 (Choked flow 방지)

    return {
        "dP_dl_fric": dP_dl_fric,
        "dP_dl_elev": dP_dl_elev,
        "E_k": E_k,
        "flow_regime": regime,
        "f_tp": f_tp,
        "H_L": H_L,
        "lambda_L": lambda_L,
        "v_m": v_m,
        "N_Fr": N_Fr,
        "Re_n": Re_n
    }

def calculate_heat_transfer(Re, Pr, k_fluid, D_i, D_o, k_pipe, k_ins, t_ins, h_o):
    """
    총괄 열전달 계수 (U) 계산
    """
    # 1. 내부 대류 (Dittus-Boelter, 냉각 가정 n=0.3)
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36
    h_i = (Nu * k_fluid) / D_i if D_i > 0 else 1e-5
    
    # 2. 열저항 계산 (외경 D_o 기준)
    R_i = D_o / (D_i * h_i)
    R_pipe = (D_o / (2 * k_pipe)) * math.log(D_o / D_i)
    
    R_ins = 0
    if t_ins > 0 and k_ins > 0:
        D_ins_o = D_o + 2 * t_ins
        R_ins = (D_o / (2 * k_ins)) * math.log(D_ins_o / D_o)
    
    R_o = 1.0 / h_o if h_o > 0 else 0
    
    U = 1.0 / (R_i + R_pipe + R_ins + R_o)
    return U

st.title("Multiphase Pipeline Pressure Drop Simulator 🚰")
st.markdown("1D Explicit Euler Discretization Model을 적용한 배관 압력/온도 강하 시뮬레이터입니다.")

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["🧪 1. Fluid Composition", "⚙️ 2. Conditions", "🏗️ 3. Pipe & Heat Transfer", "🧮 4. Numerical Setup"])

with tab1:
    st.subheader("Fluid Mixture Definition")
    selected_fluids = st.multiselect("Select Fluids (CoolProp)", AVAILABLE_FLUIDS, default=["Methane", "Ethane"])
    
    fractions = {}
    if selected_fluids:
        cols = st.columns(len(selected_fluids))
        total_frac = 0
        for i, fluid in enumerate(selected_fluids):
            val = cols[i].number_input(f"{fluid} Ratio", min_value=0.0, value=1.0 if i==0 else 0.1)
            fractions[fluid] = val
            total_frac += val
        
        if total_frac > 0:
            norm_fractions = {k: v / total_frac for k, v in fractions.items()}
            if len(norm_fractions) == 1:
                fluid_string = list(norm_fractions.keys())[0]
            else:
                fluid_string = "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])
        else:
            st.error("Total ratio must be > 0")
            fluid_string = None
    else:
        fluid_string = None

with tab2:
    st.subheader("Inlet Operating Conditions")
    col1, col2, col3 = st.columns(3)
    T_inlet_C = col1.number_input("Inlet Temperature (°C)", value=50.0)
    P_inlet_bar = col2.number_input("Inlet Pressure (bar)", value=50.0)
    mass_flow = col3.number_input("Mass Flow Rate (kg/s)", value=25.0)

with tab3:
    st.subheader("Pipe Geometry & Material")
    col1, col2, col3 = st.columns(3)
    L_total = col1.number_input("Total Length (m)", value=1000.0)
    D_inner = col2.number_input("Inner Diameter (m)", value=0.2)
    thickness = col3.number_input("Wall Thickness (m)", value=0.01)
    
    col4, col5, col6 = st.columns(3)
    elevation_angle = col4.number_input("Elevation Angle (deg, + is up)", value=0.0)
    roughness = col5.number_input("Roughness (m)", value=0.000045, format="%.6f")
    k_pipe = col6.number_input("Pipe Thermal Cond. (W/m·K)", value=45.0)
    
    st.subheader("Environment & Insulation")
    col7, col8, col9 = st.columns(3)
    T_amb_C = col7.number_input("Ambient Temp (°C)", value=15.0)
    h_ext = col8.number_input("Ext. Convection h_o (W/m²·K)", value=10.0)
    t_ins = col9.number_input("Insulation Thickness (m)", value=0.0)
    k_ins = st.number_input("Insulation Thermal Cond. (W/m·K)", value=0.035) if t_ins > 0 else 0.0

with tab4:
    st.subheader("Discretization Settings")
    increments = st.number_input("Number of Increments (Nodes)", min_value=10, max_value=1000, value=100)
    dL = L_total / increments
    st.info(f"Segment Length (ΔL) = {dL:.2f} m")

st.divider()

if st.button("🚀 압력 강하 시뮬레이션 실행 (Calculate)", type="primary", use_container_width=True):
    if not fluid_string:
        st.error("유체 조성을 먼저 설정해주세요.")
        st.stop()
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 초기 조건 세팅
    T_current_K = T_inlet_C + 273.15
    P_current_Pa = P_inlet_bar * 100000
    D_outer = D_inner + 2 * thickness
    
    results = []
    first_node_debug = {} # Glass-box용 첫 노드 데이터
    
    start_time = time.time()
    
    try:
        for i in range(int(increments)):
            status_text.text(f"계산 중... Node {i+1}/{int(increments)}")
            
            # 1. Thermodynamics Update (CoolProp)
            try:
                phase_raw = PhaseSI('T', T_current_K, 'P', P_current_Pa, fluid_string)
            except:
                phase_raw = "unknown"
                
            Q = -1.0 # 건도 (단상 기본값)
            is_twophase = False
            
            if phase_raw == 'twophase':
                is_twophase = True
                try:
                    Q = PropsSI('Q', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                except:
                    is_twophase = False # 혼합물 수렴 실패 방어
            
            # 물성치 계산 (단상 vs 이상)
            if not is_twophase:
                rho = PropsSI('D', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                try:
                    mu = PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                except ValueError: # Viscosity Fallback (Leduc's rule approx)
                    mu_sum = 0
                    for f, frac in norm_fractions.items():
                        try:
                            mu_sum += frac * PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, f)
                        except:
                            pass
                    mu = mu_sum if mu_sum > 0 else 1e-5
                
                Cp = PropsSI('C', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                k_fluid = PropsSI('L', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                
                # 유속 및 마찰계수
                velocity = mass_flow / (rho * math.pi * (D_inner/2)**2)
                Re = (rho * velocity * D_inner) / mu
                f_factor = churchill_friction_factor(Re, roughness/D_inner)
                
                # 압력 강하 (Darcy)
                dP_dl_fric = (f_factor * rho * velocity**2) / (2 * D_inner)
                dP_dl_elev = rho * 9.81 * math.sin(math.radians(elevation_angle))
                dP_dl_total = dP_dl_fric + dP_dl_elev
                
                flow_regime = "1-Phase"
                Pr = (Cp * mu) / k_fluid
                
                # 디버그 데이터 수집
                if i == 0:
                    first_node_debug = {
                        "Phase": "Single Phase", "Density": rho, "Viscosity": mu, "Velocity": velocity, "Re": Re, "f_factor": f_factor
                    }
            
            else:
                # 2상 유동 (Beggs & Brill)
                # 혼합물 특성상 Q 0, 1에서 물성치가 튈 수 있으므로 순수성분 조합 근사치 사용 (단순화)
                # 실제 고도화 앱에서는 VLE (Flash calculation) 필요. 여기서는 CoolProp의 Mixture 2-phase 한계를 Q로 우회.
                rho_L = PropsSI('D', 'P', P_current_Pa, 'Q', 0, fluid_string)
                rho_G = PropsSI('D', 'P', P_current_Pa, 'Q', 1, fluid_string)
                
                try: mu_L = PropsSI('V', 'P', P_current_Pa, 'Q', 0, fluid_string)
                except: mu_L = 1e-3 # Fallback (물 수준)
                try: mu_G = PropsSI('V', 'P', P_current_Pa, 'Q', 1, fluid_string)
                except: mu_G = 1e-5 # Fallback (가스 수준)
                
                Cp = PropsSI('C', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                try: k_fluid = PropsSI('L', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                except: k_fluid = 0.5
                
                mass_flow_G = mass_flow * Q
                mass_flow_L = mass_flow * (1 - Q)
                
                A = math.pi * (D_inner/2)**2
                v_SG = mass_flow_G / (rho_G * A)
                v_SL = mass_flow_L / (rho_L * A)
                
                bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, elevation_angle, roughness, P_current_Pa)
                
                dP_dl_total = (bb_result["dP_dl_fric"] + bb_result["dP_dl_elev"]) / (1 - bb_result["E_k"])
                flow_regime = bb_result["flow_regime"]
                
                # 혼합 물성치 근사
                Re = bb_result.get("Re_n", 1e5)
                mu_mix = bb_result.get("lambda_L", 0.5) * mu_L + (1 - bb_result.get("lambda_L", 0.5)) * mu_G
                Pr = (Cp * mu_mix) / k_fluid if k_fluid > 0 else 1.0
                
                if i == 0:
                    first_node_debug = {
                        "Phase": "Two-Phase", "Quality": Q, "v_SL": v_SL, "v_SG": v_SG, 
                        "Regime": flow_regime, "N_Fr": bb_result["N_Fr"], "Liquid Holdup (H_L)": bb_result["H_L"],
                        "Acceleration Term (E_k)": bb_result["E_k"]
                    }

            # 열전달 (dT/dL)
            U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe, k_ins, t_ins, h_ext)
            T_amb_K = T_amb_C + 273.15
            dT_dl = (U * math.pi * D_outer * (T_amb_K - T_current_K)) / (mass_flow * Cp)
            
            # 결과 저장
            results.append({
                "Node": i,
                "Length (m)": i * dL,
                "Pressure (bar)": P_current_Pa / 100000,
                "Temperature (°C)": T_current_K - 273.15,
                "Phase": "2-Phase" if Q >= 0 else "1-Phase",
                "Quality": Q if Q >= 0 else None,
                "Regime": flow_regime,
                "dP/dL (Pa/m)": dP_dl_total,
                "dT/dL (°C/m)": dT_dl
            })
            
            # 상태 업데이트 (Euler Method)
            P_current_Pa -= dP_dl_total * dL
            T_current_K += dT_dl * dL
            
            # 압력이 0 이하로 떨어지는 경우 (초크 또는 오류) 방어
            if P_current_Pa <= 10000:
                st.warning(f"🚨 경고: Node {i}에서 파이프라인 내부 압력이 거의 0에 도달했습니다. 시뮬레이션을 중단합니다. (배관이 너무 길거나 마찰이 큽니다)")
                break
                
            progress_bar.progress((i + 1) / increments)
        
        # 시뮬레이션 종료 처리
        progress_bar.empty()
        status_text.empty()
        
        df_res = pd.DataFrame(results)
        final_P = df_res.iloc[-1]['Pressure (bar)']
        final_T = df_res.iloc[-1]['Temperature (°C)']
        total_dP = P_inlet_bar - final_P
        
        st.success(f"✅ 계산 완료! (소요 시간: {time.time() - start_time:.2f}초)")
        st.header("🎯 Final Pipeline Output")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 압력 강하 (Total ΔP)", f"{total_dP:.2f} bar")
        m2.metric("출구 압력 (Exit Pressure)", f"{final_P:.2f} bar")
        m3.metric("출구 온도 (Exit Temp)", f"{final_T:.1f} °C")
        
        with st.expander("🔍 1단계 노드(Node 0) 상세 계산 과정 보기 (Glass-box)"):
            st.markdown("수치해석 1회차(Node 0 -> 1)에서 어떤 공식과 변수들이 사용되었는지 투명하게 공개합니다.")
            
            if first_node_debug.get("Phase") == "Single Phase":
                st.markdown("### 1-Phase 유동 로직 적용")
                st.latex(r"\frac{dP}{dL} = \frac{f \cdot \rho \cdot v^2}{2D} + \rho g \sin(\theta)")
                st.latex(r"f_{Churchill} = 8 \left[ \left(\frac{8}{Re}\right)^{12} + \frac{1}{(A+B)^{1.5}} \right]^{1/12}")
            else:
                st.markdown("### 2-Phase 유동 로직 적용 (Beggs & Brill 1973)")
                st.latex(r"\frac{dP}{dL} = \frac{\left(\frac{dP}{dL}\right)_{fric} + \left(\frac{dP}{dL}\right)_{elev}}{1 - E_k}")
                st.latex(r"E_k = \frac{\rho_s v_m v_{SG}}{P}")
                st.markdown("** Beggs & Brill 유동 양식(Flow Regime) 결정 트리 사용됨")
            
            st.markdown("#### 중간 계산 변수 값 (Node 0)")
            st.json(first_node_debug)

        st.header("📈 Profile Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Pressure Profile (bar)")
            st.line_chart(df_res.set_index('Length (m)')['Pressure (bar)'], color="#ff4b4b")
        with c2:
            st.markdown("#### Temperature Profile (°C)")
            st.line_chart(df_res.set_index('Length (m)')['Temperature (°C)'], color="#0068c9")
            
        st.header("📋 Detailed Discretization Table")
        st.dataframe(df_res, use_container_width=True)

    except Exception as e:
        st.error(f"계산 중 치명적 오류가 발생했습니다: {e}")
