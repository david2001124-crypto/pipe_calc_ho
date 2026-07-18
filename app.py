import streamlit as st
import pandas as pd
from CoolProp.CoolProp import PropsSI
import CoolProp

# --- 페이지 설정 ---
st.set_page_config(page_title="혼합 유체 물성치 계산기", page_icon="🧪", layout="wide")

st.title("혼합 유체 물성치 계산기 🧪")
st.markdown("CoolProp 라이브러리를 활용하여 사용자가 구성한 혼합 유체의 물성치(밀도, 점도, 비열 등)를 계산합니다.")

# 자주 사용하는 유체 리스트 (CoolProp 지원 명칭)
AVAILABLE_FLUIDS = [
    "Methane", "Ethane", "Propane", "n-Butane", "IsoButane", 
    "n-Pentane", "Isopentane", "Hexane", "Heptane",
    "Nitrogen", "CarbonDioxide", "Oxygen", "Water", "Ammonia"
]

# --- UI 레이아웃 분할 ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("1. 유체 조성 설정 (Composition)")
    
    # 멀티셀렉트로 원하는 유체 선택
    selected_fluids = st.multiselect(
        "혼합할 유체 성분을 선택하세요:", 
        AVAILABLE_FLUIDS, 
        default=["Methane", "Ethane"]
    )
    
    fractions = {}
    total_fraction = 0.0
    
    if selected_fluids:
        st.markdown("**각 성분의 몰분율(Mole Fraction) 비율을 입력하세요.** (합이 1이 아니어도 자동 정규화됩니다)")
        for fluid in selected_fluids:
            # 기본값을 1.0으로 주어 사용자가 비율(예: 8, 2)로 입력하기 쉽게 구성
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

st.divider()

# --- 계산 로직 및 결과 출력 ---
st.header("3. 물성치 계산 결과")

if st.button("물성치 계산하기", type="primary") and selected_fluids and total_fraction > 0:
    try:
        # 단위 변환
        T_K = T_C + 273.15
        P_Pa = P_bar * 100000.0
        
        # 1. 조성 정규화 (합이 1.0이 되도록)
        norm_fractions = {k: v / total_fraction for k, v in fractions.items()}
        
        # 2. CoolProp 혼합물 입력 문자열 생성
        # 형식: HEOS::Methane[0.8]&Ethane[0.2]
        if len(norm_fractions) == 1:
            fluid_string = list(norm_fractions.keys())[0] # 단일 유체
        else:
            mix_parts = [f"{fluid}[{frac:.6f}]" for fluid, frac in norm_fractions.items()]
            fluid_string = "HEOS::" + "&".join(mix_parts)
            
        # 3. 물성치 계산 (CoolProp)
        density = PropsSI('D', 'T', T_K, 'P', P_Pa, fluid_string)         # kg/m^3
        specific_heat = PropsSI('C', 'T', T_K, 'P', P_Pa, fluid_string)   # J/kg-K
        
        # 혼합물 점도의 경우 2상(2-phase) 등 특정 조건에서 에러가 날 수 있어 예외처리
        try:
            viscosity = PropsSI('V', 'T', T_K, 'P', P_Pa, fluid_string)   # Pa-s
            viscosity_cp = viscosity * 1000 # cP 변환
        except ValueError:
            viscosity = None
            
        try:
            enthalpy = PropsSI('H', 'T', T_K, 'P', P_Pa, fluid_string) / 1000 # kJ/kg
        except:
            enthalpy = None

        volume_flow = flow_rate / density if density > 0 else 0 # m^3/s

        # 4. 결과 출력
        st.success("계산이 성공적으로 완료되었습니다!")
        
        # 정규화된 조성을 보여주는 데이터프레임
        st.markdown("##### 적용된 혼합 조성비 (Normalized Mole Fractions)")
        df_comp = pd.DataFrame([norm_fractions]).T.reset_index()
        df_comp.columns = ["Component", "Mole Fraction"]
        df_comp["Mole Fraction (%)"] = (df_comp["Mole Fraction"] * 100).map("{:.2f} %".format)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)
        
        # 물성치 메트릭 출력
        st.markdown("##### 혼합물 열역학 상태값")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        mcol1.metric("밀도 (Density)", f"{density:,.2f} kg/m³")
        
        if viscosity:
            mcol2.metric("점도 (Viscosity)", f"{viscosity_cp:,.3f} cP")
        else:
            mcol2.metric("점도 (Viscosity)", "계산 불가 (Phase Error)")
            
        mcol3.metric("비열 (Cp)", f"{specific_heat:,.1f} J/kg·K")
        mcol4.metric("체적 유량 (Vol Flow)", f"{volume_flow:,.3f} m³/s")
        
    except Exception as e:
        st.error("물성치 계산 중 오류가 발생했습니다. 입력한 온도/압력에서 해당 조성이 유효하지 않은 상(Phase) 상태일 수 있습니다.")
        st.error(f"상세 에러 내용: {e}")
