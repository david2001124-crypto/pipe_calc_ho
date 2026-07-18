import streamlit as st
import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
import CoolProp

# --- 페이지 설정 ---
st.set_page_config(page_title="혼합 유체 물성치 계산기", page_icon="🧪", layout="wide")

st.title("혼합 유체 물성치 계산기 🧪")
st.markdown("CoolProp 라이브러리를 활용하여 사용자가 구성한 혼합 유체의 상태(Phase)와 물성치를 계산합니다.")

# 1. CoolProp에서 지원하는 모든 유체 리스트 동적 로드
@st.cache_data
def get_all_fluids():
    try:
        fluids_str = get_global_param_string('FluidsList')
        return sorted(fluids_str.split(','))
    except:
        # 로드 실패 시 자주 쓰이는 유체 기본값 제공
        return ["Methane", "Ethane", "Propane", "Nitrogen", "CarbonDioxide", "Water"]

AVAILABLE_FLUIDS = get_all_fluids()

# --- UI 레이아웃 분할 ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("1. 유체 조성 설정 (Composition)")
    
    selected_fluids = st.multiselect(
        "혼합할 유체 성분을 선택하세요 (CoolProp 전체 유체 지원):", 
        AVAILABLE_FLUIDS, 
        default=["Methane", "Ethane"]
    )
    
    fractions = {}
    total_fraction = 0.0
    
    if selected_fluids:
        st.markdown("**각 성분의 몰분율(Mole Fraction) 비율을 입력하세요.** (합이 1이 아니어도 자동 정규화됩니다)")
        for fluid in selected_fluids:
            val = st.number_input(f"{fluid} 비율", min_value=0.0, value=1.0, step=0.1)
            fractions[fluid] = val
            total_fraction += val
            
        if total_fraction == 0:
            st.warning("유체 비율의 합은 0보다 커야 합니다.")
    else:
        st.info("유체를 하나 이상 선택해 주세요.")

with col2:
    st.header("2. 운전 조건 입력 (Conditions)")
    
    col_t, col_p = st.columns(2)
    with col_t:
        T_C = st.number_input("온도 (°C)", value=25.0, step=1.0)
    with col_p:
        P_bar = st.number_input("압력 (bar)", value=10.0, step=0.5)
        
    flow_rate = st.number_input("질량 유량 (kg/s) - 옵션", value=10.0, step=1.0)
    
    st.caption("⚠️ **순수 유체(단일 성분)의 2상 구역 주의사항**\n단일 유체는 2상(기액 혼합) 구간에서 온도와 압력이 종속적이므로, 포화 구간 내에서 온도와 압력을 동시에 입력하여 계산하면 에러가 발생할 수 있습니다. (혼합물은 정상 계산됨)")

st.divider()

# --- 계산 로직 및 결과 출력 ---
st.header("3. 상태 판별 및 물성치 계산 결과")

if st.button("물성치 계산하기", type="primary") and selected_fluids and total_fraction > 0:
    try:
        # 단위 변환
        T_K = T_C + 273.15
        P_Pa = P_bar * 100000.0
        
        # 조성 정규화
        norm_fractions = {k: v / total_fraction for k, v in fractions.items()}
        
        # CoolProp 입력 문자열 생성
        if len(norm_fractions) == 1:
            fluid_string = list(norm_fractions.keys())[0] # 단일 유체
        else:
            mix_parts = [f"{fluid}[{frac:.6f}]" for fluid, frac in norm_fractions.items()]
            fluid_string = "HEOS::" + "&".join(mix_parts)
            
        # --- 1) 상(Phase) 판별 ---
        try:
            phase_raw = PhaseSI('T', T_K, 'P', P_Pa, fluid_string)
            phase_map = {
                'liquid': '단상 액체 (Subcooled Liquid)',
                'gas': '단상 기체 (Superheated Vapor)',
                'twophase': '2상 혼합 상태 (Two-Phase)',
                'supercritical': '초임계 상태 (Supercritical)',
                'supercritical_gas': '초임계 기체 (Supercritical Gas)',
                'supercritical_liquid': '초임계 액체 (Supercritical Liquid)'
            }
            phase_display = phase_map.get(phase_raw, phase_raw)
        except:
            phase_raw = "unknown"
            phase_display = "판별 불가 (조건 범위를 벗어났거나 계산 오류)"

        # --- 2) 건도(Quality) 계산 (2상일 경우) ---
        quality = None
        if phase_raw == 'twophase':
            try:
                quality = PropsSI('Q', 'T', T_K, 'P', P_Pa, fluid_string)
            except:
                pass

        # UI 출력: 상 상태 (Phase State)
        st.markdown("##### 🌡️ 유체 상 상태 (Phase State)")
        if phase_raw == 'twophase' and quality is not None:
            st.info(f"현재 조건에서 유체는 **{phase_display}**입니다.\n\n💧 **건도(Quality, 기상 질량비): {quality:.4f}** (기체 {quality*100:.1f}%, 액체 {(1-quality)*100:.1f}%)")
        else:
            st.info(f"현재 조건에서 유체는 **{phase_display}**입니다.")

        # --- 3) 열역학 물성치 계산 ---
        density = PropsSI('D', 'T', T_K, 'P', P_Pa, fluid_string)         # kg/m^3
        specific_heat = PropsSI('C', 'T', T_K, 'P', P_Pa, fluid_string)   # J/kg-K
        
        # 점도 예외 처리
        try:
            viscosity = PropsSI('V', 'T', T_K, 'P', P_Pa, fluid_string)   # Pa-s
            viscosity_cp = viscosity * 1000 # cP 변환
        except ValueError:
            viscosity = None

        volume_flow = flow_rate / density if density > 0 else 0 # m^3/s

        # 물성치 메트릭 출력
        st.markdown("##### 📊 혼합물 물성치 (Properties)")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("밀도 (Density)", f"{density:,.2f} kg/m³")
        if viscosity:
            mcol2.metric("점도 (Viscosity)", f"{viscosity_cp:,.3f} cP")
        else:
            mcol2.metric("점도 (Viscosity)", "계산 불가")
        mcol3.metric("비열 (Cp)", f"{specific_heat:,.1f} J/kg·K")
        mcol4.metric("체적 유량 (Vol Flow)", f"{volume_flow:,.3f} m³/s")
        
        # 정규화된 조성을 보여주는 표
        st.markdown("##### 🧪 적용된 혼합 조성비 (Normalized Mole Fractions)")
        df_comp = pd.DataFrame([norm_fractions]).T.reset_index()
        df_comp.columns = ["Component", "Mole Fraction"]
        df_comp["Mole Fraction (%)"] = (df_comp["Mole Fraction"] * 100).map("{:.2f} %".format)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)
        
    except Exception as e:
        st.error("물성치 계산 중 오류가 발생했습니다. 입력한 온도/압력에서 해당 조성이 지원되지 않거나 수렴하지 않는 상태일 수 있습니다.")
        st.error(f"상세 에러 내용: {e}")
