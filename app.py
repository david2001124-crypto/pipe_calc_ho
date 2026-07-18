from CoolProp.CoolProp import PropsSI

def get_fluid_properties(T_kelvin, P_pascal, fluid_name="Methane"):
    # 1. 밀도 (kg/m³) - Density
    density = PropsSI('D', 'T', T_kelvin, 'P', P_pascal, fluid_name)
    
    # 2. 점도 (Pa·s) - Viscosity
    viscosity = PropsSI('V', 'T', T_kelvin, 'P', P_pascal, fluid_name)
    
    # 3. 등압 비열 (J/kg·K) - Specific heat
    cp = PropsSI('C', 'T', T_kelvin, 'P', P_pascal, fluid_name)
    
    # (참고) 이상 유동 시 기상/액상 물성치도 'Q'(건도) 값을 통해 분리 추출 가능
    
    return density, viscosity, cp

# 사용 예시 (섭씨 25도, 10 bar 메탄)
rho, mu, cp = get_fluid_properties(298.15, 1000000, "Methane")
