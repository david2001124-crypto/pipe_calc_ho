# ... existing code ...
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

# STREAMING_CHUNK:Rendering dynamic component builder...
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
    f_type = fc1.selectbox("피팅/밸브 종류", list(material_db.FITTING_LE_D_DB.keys()))
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
# ... existing code ...
        for comp_idx, comp in enumerate(st.session_state.pipeline):
            if comp.get("type") == "Pipe":
                # 파이프 컴포넌트: 실제 길이가 존재하며, Node(격자)를 나누어 계산
                curr_D_inner = comp.get("D_inner", curr_D_inner)
                curr_thickness = comp.get("thickness", curr_thickness)
                if "material" in comp:
                    curr_mat_info = material_db.MATERIAL_MAP[comp["material"]]
                curr_roughness = curr_mat_info["roughness_m"]
                asme_table = material_db.RAW_DB[curr_mat_info["asme_category"]][curr_mat_info["asme_grade"]]
                D_outer = curr_D_inner + 2 * curr_thickness
                A_cross = math.pi * (curr_D_inner / 2)**2
                
                length = comp.get("length", 10.0)
                dL = length / N_per_pipe if N_per_pipe > 0 else 0
                
                if comp.get("elev_type", "Angle (deg)") == "Height (m)":
                    angle_deg = 0 if length == 0 else math.degrees(math.asin(max(min(comp.get("elev_val", 0.0) / length, 1.0), -1.0)))
                else:
                    angle_deg = comp.get("elev_val", 0.0)
                dZ = dL * math.sin(math.radians(angle_deg))

                for i in range(N_per_pipe):
                    if P_current_Pa < 10000: # 0.1 bar 이하로 압력이 떨어질 경우 계산 중지 (진공 상태 방지)
                        raise ValueError(f"[{comp_idx+1}번 배관] 내부 유체 압력이 {P_current_Pa/1e5:.3f} bar로 너무 낮습니다. 유량이 과다하거나 배관 직경이 너무 작아 마찰 손실이 극심합니다.")
                        
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
                    
                    try:
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
                    except Exception as e:
                        raise ValueError(f"[{comp_idx+1}번 구간] 해당 온도/압력에서 유체 밀도 등 주요 물성치 계산 실패 (P={P_current_Pa/1e5:.2f}bar). 2상 혼합 범위를 벗어났을 수 있습니다.")

                    U = calculate_heat_transfer(Re, Pr, k_fluid, curr_D_inner, D_outer, k_pipe_current, k_ins, t_ins, h_ext)
# ... existing code ...
            elif comp.get("type") == "Fitting":
                # 피팅/밸브 컴포넌트
                f_name = comp.get("name", "Unknown")
                qty = comp.get("qty", 1)
                
                if P_current_Pa < 10000: # 0.1 bar 이하 검사
                        raise ValueError(f"[{comp_idx+1}번 밸브] 통과 전 압력이 {P_current_Pa/1e5:.3f} bar로 너무 낮습니다.")

                # 직전 파이프의 직경(curr_D_inner) 상속
                L_e_total = (material_db.FITTING_LE_D_DB.get(f_name, 30) * curr_D_inner) * qty
# ... existing code ...
    with st.spinner("🤖 인공지능 Auto-Mesh 수렴 계산 중..."):
        N_nodes = 5
        max_iter = 6
        prev_P, prev_T = None, None
        converged = False
        
        try:
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
                
        except ValueError as ve:
            st.error(f"🚨 시뮬레이션 중단: {ve}")
            st.stop()
        except Exception as e:
            st.error(f"🚨 예상치 못한 시스템 오류 발생: {e}")
            st.stop()

    if global_audit_tracker:
# ... existing code ...
```eof
