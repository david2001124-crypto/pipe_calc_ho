import streamlit as st

st.set_page_config(page_title="파이프 압력 강하 계산기", page_icon="🚰")

st.title("파이프 압력 강하 계산기 🚰")
st.markdown("Darcy-Weisbach 방정식을 이용해 배관 내 유체의 압력 강하를 계산합니다.")

# 사이드바 입력창 구성
st.sidebar.header("입력 변수 (Input Parameters)")
length = st.sidebar.number_input("파이프 길이 $L$ (m)", value=100.0, step=1.0)
diameter = st.sidebar.number_input("파이프 내경 $D$ (m)", value=0.1, step=0.01)
velocity = st.sidebar.number_input("유속 $V$ (m/s)", value=2.0, step=0.1)
density = st.sidebar.number_input("유체 밀도 $\\rho$ (kg/m³)", value=1000.0, step=10.0) # 기본값: 물
friction_factor = st.sidebar.number_input("마찰 계수 $f$", value=0.02, step=0.001)

# 계산 로직 및 화면 출력
if st.button("압력 강하 계산하기", type="primary"):
    # 계산식: dP = f * (L/D) * (rho * V^2 / 2)
    pressure_drop_pa = friction_factor * (length / diameter) * (density * velocity**2 / 2)
    pressure_drop_bar = pressure_drop_pa / 100000  # Pa 단위를 bar 단위로 변환

    st.success("계산이 완료되었습니다!")
    
    # 결과값을 깔끔한 카드 형태로 출력
    col1, col2 = st.columns(2)
    col1.metric(label="압력 강하 (Pa)", value=f"{pressure_drop_pa:,.2f} Pa")
    col2.metric(label="압력 강하 (bar)", value=f"{pressure_drop_bar:,.4f} bar")