import streamlit as st
import pandas as pd
import numpy as np
import math
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import time

st.set_page_config(page_title="Multiphase Pipeline Simulator V2", page_icon="🚰", layout="wide")

@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

MATERIAL_ROUGHNESS = {
    "Carbon Steel / Wrought Iron (0.0675 mm)": 0.0000675,
    "Weld Steel (0.045 mm)": 0.000045,
    "Stainless Steel, bead blasted (0.0035 mm)": 0.0000035,
    "Copper-Nickel / Drawn (0.0015 mm)": 0.0000015,
    "GRE / Plastic (0.00425 mm)": 0.00000425,
    "Galvanized Steel (0.15 mm)": 0.00015,
    "Rusted Steel / Corrosion (2.075 mm)": 0.002075
}

def churchill_friction_factor(Re, e_D):
    """
    Churchill (1977) equation for Darcy friction factor.
    """
    if Re < 1e-10: return 0.0
    A = (2.457 * math.log(1.0 / ((7.0 / Re)**0.9 + 0.27 * e_D)))**16
    B = (37530.0 / Re)**16
    f = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5)**(1/12.0)
    return f

def calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D, theta_deg, roughness, P_Pa):
    """
    Beggs and Brill (1973) Multiphase Flow Model
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
        if regime == "Segregated": C_corr = (1 - lambda_L) * math.log(lambda_L**2 * N_Fr * v_SL)
        elif regime == "Intermittent": C_corr = (1 - lambda_L) * math.log(lambda_L**0.1 * N_Fr * v_SL)
    
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
    Nu = 0.023 * (Re**0.8) * (Pr**0.3) if Re > 2300 else 4.36
    h_i = (Nu * k_fluid) / D_i if D_i > 0 else 1e-5
    
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
st.markdown("1D Explicit Euler Discretization Model을 적용한 배관 압력/온도 강하 시뮬레이터 (V2)")
st.info("모든 값은 필수 입력 사항입니다. 누락된 값이 있을 경우 계산이 진행되지 않습니다.")

# 1. Fluid Composition
st.header("1. Fluid Composition")
selected_fluids = st.multiselect("Select Fluids (CoolProp)", AVAILABLE_FLUIDS, default=None)
fractions = {}
if selected_fluids:
    cols = st.columns(len(selected_fluids))
    for i, fluid in enumerate(selected_fluids):
        val = cols[i].number_input(f"{fluid} Ratio (Mole fraction)", value=None, step=None, format="%.4f")
        fractions[fluid] = val

# 2. Operating Conditions
st.header("2. Operating Conditions")
c1, c2, c3 = st.columns(3)
T_inlet_C = c1.number_input("Inlet Temperature (°C)", value=None, step=None, format="%.4f")
P_inlet_bar = c2.number_input("Inlet Pressure (bar)", value=None, step=None, format="%.4f")
mass_flow = c3.number_input("Mass Flow Rate (kg/s)", value=None, step=None, format="%.4f")

# 3. Pipe Geometry & Material
st.header("3. Pipe Geometry & Material")
c4, c5, c6 = st.columns(3)
L_total = c4.number_input("Total Length (m)", value=None, step=None, format="%.4f")
elev_type = c5.selectbox("Elevation Type", ["Angle (deg)", "Height (m)"])
elev_val = c6.number_input(f"Elevation {elev_type}", value=None, step=None, format="%.4f")

c7, c8 = st.columns(2)
D_inner = c7.number_input("Inner Diameter (m)", value=None, step=None, format="%.4f")
thickness = c8.number_input("Wall Thickness (m)", value=None, step=None, format="%.4f")

st.markdown("**(Reference) Pipe Roughness Source:** [Engineering ToolBox](https://www.engineeringtoolbox.com/surface-roughness-ventilation-ducts-d_209.html#gsc.tab=0)")
mat_keys = list(MATERIAL_ROUGHNESS.keys())
selected_mat = st.selectbox("Pipe Material (determines absolute roughness)", [None] + mat_keys)

c9, c10, c11 = st.columns(3)
k_pipe = c9.number_input("Pipe Thermal Cond. (W/m·K)", value=None, step=None, format="%.4f")
t_ins = c10.number_input("Insulation Thickness (m)", value=None, step=None, format="%.4f", help="보온재가 없다면 0을 입력하세요.")
k_ins = c11.number_input("Insulation Thermal Cond. (W/m·K)", value=None, step=None, format="%.4f", help="두께가 0이면 무시됩니다.")

# 4. Environment
st.header("4. Environment")
c12, c13 = st.columns(2)
T_amb_C = c12.number_input("Ambient Temp (°C)", value=None, step=None, format="%.4f")
h_ext = c13.number_input("Ext. Convection h_o (W/m²·K)", value=None, step=None, format="%.4f")

# 5. Numerical Setup
st.header("5. Numerical Setup")
increments = st.number_input("Number of Increments (Nodes)", value=None, step=None, format="%d")

st.divider()

if st.button("🚀 압력 강하 시뮬레이션 실행 (Calculate)", type="primary", use_container_width=True):
    
    # Validation Logic
    missing_fields = []
    
    if not selected_fluids: missing_fields.append("Fluid Selection")
    missing_fracs = [f for f, val in fractions.items() if val is None]
    if missing_fracs: missing_fields.append(f"Mole fractions for: {', '.join(missing_fracs)}")
    
    core_inputs = {
        "Inlet Temperature": T_inlet_C, "Inlet Pressure": P_inlet_bar, "Mass Flow Rate": mass_flow,
        "Total Length": L_total, "Elevation Value": elev_val, "Inner Diameter": D_inner,
        "Wall Thickness": thickness, "Pipe Material": selected_mat, "Pipe Thermal Cond.": k_pipe,
        "Insulation Thickness": t_ins, "Ambient Temp": T_amb_C, "Ext. Convection": h_ext,
        "Number of Increments": increments
    }
    
    for k, v in core_inputs.items():
        if v is None: missing_fields.append(k)
        
    if t_ins is not None and t_ins > 0 and k_ins is None:
        missing_fields.append("Insulation Thermal Cond. (k_ins is required if thickness > 0)")
        
    if missing_fields:
        st.error(f"🚨 다음 필수 입력값이 누락되었습니다. 모두 채워주세요: \n\n**{', '.join(missing_fields)}**")
        st.stop()
        
    # Validation Pass - Setup logic
    total_frac = sum(fractions.values())
    if total_frac <= 0:
        st.error("총 조성 비율 합계는 0보다 커야 합니다.")
        st.stop()
        
    norm_fractions = {k: v / total_frac for k, v in fractions.items()}
    if len(norm_fractions) == 1:
        fluid_string = list(norm_fractions.keys())[0]
    else:
        fluid_string = "HEOS::" + "&".join([f"{f}[{frac}]" for f, frac in norm_fractions.items()])

    # Elevation logic
    if elev_type == "Height (m)":
        if L_total == 0 or abs(elev_val) > L_total:
            st.error("높이(Height)는 배관 전체 길이(Length)보다 클 수 없습니다.")
            st.stop()
        elevation_angle = math.degrees(math.asin(elev_val / L_total))
    else:
        elevation_angle = elev_val

    roughness = MATERIAL_ROUGHNESS[selected_mat]
    dL = L_total / increments
    
    T_current_K = T_inlet_C + 273.15
    P_current_Pa = P_inlet_bar * 100000
    D_outer = D_inner + 2 * thickness
    t_ins_calc = t_ins if t_ins is not None else 0.0
    k_ins_calc = k_ins if k_ins is not None else 0.0
    
    results = []
    first_node_debug = {}
    start_time = time.time()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(int(increments)):
            status_text.text(f"계산 중... Node {i+1}/{int(increments)}")
            
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
                try:
                    mu = PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, fluid_string)
                except ValueError: 
                    mu_sum = 0
                    for f, frac in norm_fractions.items():
                        try: mu_sum += frac * PropsSI('V', 'T', T_current_K, 'P', P_current_Pa, f)
                        except: pass
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
                    first_node_debug = {
                        "Phase": "Single Phase", "Density (kg/m3)": f"{rho:.2f}", 
                        "Viscosity (Pa.s)": f"{mu:.6e}", "Velocity (m/s)": f"{velocity:.2f}", 
                        "Re": f"{Re:.0f}", "f_factor": f"{f_factor:.5f}"
                    }
            
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
                
                mass_flow_G = mass_flow * Q
                mass_flow_L = mass_flow * (1 - Q)
                
                A = math.pi * (D_inner/2)**2
                v_SG = mass_flow_G / (rho_G * A)
                v_SL = mass_flow_L / (rho_L * A)
                
                bb_result = calculate_beggs_brill(v_SL, v_SG, rho_L, rho_G, mu_L, mu_G, D_inner, elevation_angle, roughness, P_current_Pa)
                
                dP_dl_total = (bb_result["dP_dl_fric"] + bb_result["dP_dl_elev"]) / (1 - bb_result["E_k"])
                flow_regime = bb_result["flow_regime"]
                
                Re = bb_result.get("Re_n", 1e5)
                mu_mix = bb_result.get("lambda_L", 0.5) * mu_L + (1 - bb_result.get("lambda_L", 0.5)) * mu_G
                Pr = (Cp * mu_mix) / k_fluid if k_fluid > 0 else 1.0
                
                if i == 0:
                    first_node_debug = {
                        "Phase": "Two-Phase", "Quality (Q)": f"{Q:.4f}", "v_SL (m/s)": f"{v_SL:.2f}", 
                        "v_SG (m/s)": f"{v_SG:.2f}", "Regime": flow_regime, "N_Fr": f"{bb_result['N_Fr']:.4f}", 
                        "Liquid Holdup (H_L)": f"{bb_result['H_L']:.4f}", "Accel Term (E_k)": f"{bb_result['E_k']:.6f}"
                    }

            U = calculate_heat_transfer(Re, Pr, k_fluid, D_inner, D_outer, k_pipe, k_ins_calc, t_ins_calc, h_ext)
            T_amb_K = T_amb_C + 273.15
            dT_dl = (U * math.pi * D_outer * (T_amb_K - T_current_K)) / (mass_flow * Cp)
            
            results.append({
                "Node": i,
                "Length (m)": i * dL,
                "Pressure (bar)": P_current_Pa / 100000,
                "Temperature (°C)": T_current_K - 273.15,
                "Phase": "2-Phase" if Q >= 0 else "1-Phase",
                "Quality": Q if Q >= 0 else 0,
                "Regime": flow_regime,
                "dP/dL (Pa/m)": dP_dl_total,
                "dT/dL (°C/m)": dT_dl
            })
            
            P_current_Pa -= dP_dl_total * dL
            T_current_K += dT_dl * dL
            
            if P_current_Pa <= 10000:
                st.warning(f"🚨 경고: Node {i}에서 압력이 0에 도달했습니다. 배관이 너무 길거나 마찰이 극심합니다.")
                break
                
            progress_bar.progress((i + 1) / increments)
            
        progress_bar.empty()
        status_text.empty()
        
        df_res = pd.DataFrame(results)
        final_P = df_res.iloc[-1]['Pressure (bar)']
        final_T = df_res.iloc[-1]['Temperature (°C)']
        total_dP = P_inlet_bar - final_P
        
        st.success(f"✅ 시뮬레이션 완료! (소요 시간: {time.time() - start_time:.2f}초)")
        
        # 1_final_result_top
        st.header("🎯 Final Pipeline Output")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 압력 강하 (Total ΔP)", f"{total_dP:.4f} bar")
        m2.metric("출구 압력 (Exit Pressure)", f"{final_P:.4f} bar")
        m3.metric("출구 온도 (Exit Temp)", f"{final_T:.4f} °C")
        
        # 2_step_by_step_transparency
        with st.expander("🔍 1단계 노드(Node 0) 상세 계산 과정 보기 (Glass-box)"):
            st.markdown("수치해석 1회차(Node 0 -> 1)에서 적용된 논리와 수식을 투명하게 공개합니다.")
            if first_node_debug.get("Phase") == "Single Phase":
                st.markdown("### 1-Phase 유동 판별됨")
                st.latex(r"\frac{dP}{dL} = \frac{f \cdot \rho \cdot v^2}{2D} + \rho g \sin(\theta)")
                st.latex(r"f_{Churchill} = 8 \left[ \left(\frac{8}{Re}\right)^{12} + \frac{1}{(A+B)^{1.5}} \right]^{1/12}")
            else:
                st.markdown("### 2-Phase 혼합 유동 판별됨 (Beggs & Brill 1973)")
                st.latex(r"\frac{dP}{dL} = \frac{\left(\frac{dP}{dL}\right)_{fric} + \left(\frac{dP}{dL}\right)_{elev}}{1 - E_k}")
                st.latex(r"E_k = \frac{\rho_s v_m v_{SG}}{P}")
                st.markdown("** Beggs & Brill 유동 양식 결정 로직 및 Holdup 보정이 적용되었습니다.")
            
            st.markdown("#### 중간 계산 변수 값 (Node 0)")
            st.json(first_node_debug)

        # 4_visualizations
        st.header("📈 Profile Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Pressure Profile (bar)")
            st.line_chart(df_res.set_index('Length (m)')['Pressure (bar)'], color="#ff4b4b")
        with c2:
            st.markdown("#### Temperature Profile (°C)")
            st.line_chart(df_res.set_index('Length (m)')['Temperature (°C)'], color="#0068c9")
            
        # 3_data_table
        st.header("📋 Detailed Discretization Table")
        st.dataframe(df_res, use_container_width=True)

    except Exception as e:
        st.error(f"계산 중 치명적 오류가 발생했습니다: {e}")
