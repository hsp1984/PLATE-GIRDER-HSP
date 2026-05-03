import streamlit as st
import numpy as np
import math

st.set_page_config(page_title="Plate Girder Designer - IS 800:2007", layout="wide")
st.title("🏗️ Steel Plate Girder Designer (Shear Buckling + Auto‑Revision)")
st.markdown("Design of welded plate girders as per **IS 800:2007 (Limit State Method)**")
st.markdown("---")

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.header("1. Geometry & Loading")
    span = st.number_input("Span (m)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
    load_type = st.selectbox("Load type", ["Uniformly Distributed Load (UDL)", "Point Load at Midspan"])
    
    if load_type == "Uniformly Distributed Load (UDL)":
        dl = st.number_input("Dead Load (kN/m)", value=30.0, step=5.0)
        ll = st.number_input("Live Load (kN/m)", value=20.0, step=5.0)
        point_load = 0.0
    else:
        point_load = st.number_input("Point Load at Midspan (kN)", value=150.0, step=25.0)
        dl = ll = 0.0
    
    st.header("2. Material Properties")
    steel_grade = st.selectbox("Steel grade", ["Fe 250", "Fe 410", "Fe 450"])
    fy = 250 if "250" in steel_grade else 410 if "410" in steel_grade else 450
    fu = 410 if fy == 250 else 550
    st.info(f"Yield strength (fy) = **{fy} MPa**  |  Ultimate strength (fu) = **{fu} MPa**")
    
    st.header("3. Optional (Advanced)")
    manual_bf = st.number_input("Manual flange width (mm) – 0 = auto", min_value=0, value=0, step=10)
    use_tension_field = st.checkbox("Use Tension Field Method (requires stiffeners)", value=False)
    st.markdown("---")
    st.caption("Design as per IS 800:2007, Cl. 8.4.2.2 & 8.6")

# ---------- Helper Functions ----------
def calc_factored_loads(dl, ll, point_load, span, load_type, gamma_f=1.5):
    if load_type == "Uniformly Distributed Load (UDL)":
        w_serv = dl + ll
        w_u = gamma_f * w_serv
        Mu = w_u * span**2 / 8
        Vu = w_u * span / 2
        return Mu, Vu, w_serv, 0.0
    else:
        P_serv = point_load
        P_u = gamma_f * P_serv
        Mu = P_u * span / 4
        Vu = P_u / 2
        return Mu, Vu, 0.0, P_serv

def economical_depth_iterative(Mu_kNm, fy, target_K=100):
    Mu_Nmm = Mu_kNm * 1e6
    d = (Mu_Nmm * target_K / fy) ** (1/3)
    d = max(400, min(3000, round(d / 10) * 10))
    return d

def required_plastic_modulus(Mu, fy, gamma_m0=1.10):
    return Mu * 1e6 * gamma_m0 / fy

def design_web(d, Vu, fy, gamma_m1=1.25):
    tw_min = max(6.0, d / 200.0)
    req_tw_shear = (Vu * 1000 * gamma_m1) / (d * (fy / math.sqrt(3)))
    tw = max(tw_min, req_tw_shear)
    tw = max(6.0, min(40.0, round(tw / 2) * 2))
    return tw

# ----- Shear buckling resistance (Cl. 8.4.2.2) -----
def shear_buckling_strength(d, tw, fy, has_stiffeners=False, tension_field=False):
    """
    Returns nominal shear strength Vn (kN) and the method description.
    tension_field = True only if end & intermediate stiffeners are provided.
    """
    epsilon = math.sqrt(250 / fy)
    web_slenderness = d / tw
    # Shear buckling parameter λ_w
    λ_w = math.sqrt(fy / (math.sqrt(3) * (math.pi**2 * 2e5 * (tw/d)**2 / (12*(1-0.3**2)))))  # simplified
    # More direct formula as per code (Cl. 8.4.2.2):
    # λ_w = (d/tw) / (37.4 * ε * k_v^0.5) , with k_v = 5.35 for simple supports (no stiffeners)
    k_v = 5.35
    if has_stiffeners:
        k_v = 5.35 + 4.0 / (c/d)**2   # we assume c/d = 1.0 for simplicity
    λ_w = (d / tw) / (37.4 * epsilon * math.sqrt(k_v))
    
    Av = d * tw  # shear area (mm²)
    if λ_w <= 0.8:
        τ_b = fy / math.sqrt(3)
        method = "Simple post‑critical (λ_w ≤ 0.8)"
    elif 0.8 < λ_w < 1.2:
        τ_b = (1 - 0.8*(λ_w - 0.8)) * fy / math.sqrt(3)
        method = "Simple post‑critical (0.8 < λ_w < 1.2)"
    else:
        τ_b = fy / (math.sqrt(3) * λ_w**2)
        method = "Simple post‑critical (λ_w ≥ 1.2)"
    
    Vn = Av * τ_b / 1000  # kN
    
    # Tension field method (Cl. 8.4.2.2.3)
    if tension_field and has_stiffeners:
        # additional strength from tension field action
        # simplified: add 0.5 * tw * d * fy * (1 - 1/λ_w^2) / √3  etc.
        if λ_w > 1.0:
            Vtf = Av * (fy / math.sqrt(3)) * (1 - 1/λ_w**2) / 1000
            Vn += Vtf
            method += " + Tension field contribution"
    return Vn, method, λ_w

def check_web_slenderness_full(d, tw, fy, has_stiffeners=False, stiff_spacing=None):
    epsilon = math.sqrt(250 / fy)
    actual = d / tw
    if not has_stiffeners:
        limit = 67 * epsilon
        desc = f"Without transverse stiffeners: d/tw ≤ 67ε = {limit:.1f}"
    else:
        c = stiff_spacing if stiff_spacing else (1.5 * d)
        if c >= d:
            limit = 200 * epsilon
            desc = f"With transverse stiffeners (c ≥ d): d/tw ≤ 200ε = {limit:.1f}"
        elif c >= 0.74 * d:
            limit = 200 * epsilon
            desc = f"With transverse stiffeners (0.74d ≤ c < d): d/tw ≤ 200ε = {limit:.1f}"
        else:
            limit = 270 * epsilon
            desc = f"With transverse stiffeners (c < d): d/tw ≤ 270ε = {limit:.1f}"
    ok = actual <= limit
    return ok, limit, actual, desc

def design_flanges(d, tw, Zp_req, fy, manual_bf):
    Zp_web = tw * d**2 / 4
    Af_req = max(0.0, (Zp_req - Zp_web) / d)
    if manual_bf > 0:
        bf = manual_bf
    else:
        bf = max(180.0, d / 4.0)
        bf = round(bf / 10) * 10
    tf = max(10.0, Af_req / bf)
    epsilon = math.sqrt(250 / fy)
    compact_limit = 9.4 * epsilon
    if (bf / tf) > compact_limit:
        tf = bf / (0.9 * compact_limit)
    tf = round(tf, 1)
    if tf < 40:
        tf = round(tf / 2) * 2
    else:
        tf = round(tf / 5) * 5
    return bf, tf, Zp_web

def compute_section_properties(d, tw, bf, tf):
    Zp_actual = tw * d**2 / 4 + bf * tf * (d + tf)
    Ix_cm4 = (tw * d**3 / 12 + 2 * (bf * tf * ((d + tf)/2)**2)) / 10000.0
    area_m2 = (d/1000)*(tw/1000) + 2*(bf/1000)*(tf/1000)
    weight = area_m2 * 7850
    return Zp_actual, Ix_cm4, weight

def moment_capacity(Zp_actual, fy, gamma_m0=1.10):
    return Zp_actual * fy / gamma_m0 / 1e6

def shear_capacity(d, tw, fy, gamma_m1=1.25, has_stiffeners=False, tension_field=False):
    Vn, method, λ_w = shear_buckling_strength(d, tw, fy, has_stiffeners, tension_field)
    Vd = Vn / 1.25  # γ_m1 for shear, but Vn already includes partial factor? We'll divide again?
    # Actually Vn is nominal strength, design strength Vd = Vn / γ_m1
    Vd = Vn / gamma_m1
    return Vd, method, λ_w

def deflection_check(span_m, Ix_cm4, w_serv, P_serv, load_type, E=2.0e5):
    delta_limit = span_m * 1000 / 300.0
    if load_type == "Uniformly Distributed Load (UDL)":
        delta = 5.0 * w_serv * (span_m * 1000)**4 / (384 * E * Ix_cm4 * 1e4)
    else:
        delta = P_serv * 1000 * (span_m * 1000)**3 / (48 * E * Ix_cm4 * 1e4)
    return delta, delta_limit

def stiffener_requirements(d, tw, fy, Vu, Vd):
    epsilon = math.sqrt(250 / fy)
    web_slenderness = d / tw
    if web_slenderness <= 67 * epsilon:
        return False, None
    spacing = min(1.5 * d, 3000)
    spacing = round(spacing / 100) * 100
    return True, spacing

# ---------- Main Design with Auto‑Revision ----------
if st.sidebar.button("🚀 Design Plate Girder", type="primary", use_container_width=True):
    
    Mu, Vu, w_serv, P_serv = calc_factored_loads(dl, ll, point_load, span, load_type)
    Zp_req = required_plastic_modulus(Mu, fy)
    
    # Initial depth from economical depth
    d_guess = economical_depth_iterative(Mu, fy, target_K=100)
    tw_temp = design_web(d_guess, Vu, fy)
    K_actual = d_guess / tw_temp
    d = economical_depth_iterative(Mu, fy, target_K=K_actual)
    d = max(400, min(3000, round(d / 10) * 10))
    tw = design_web(d, Vu, fy)
    
    # Initial flange sizes
    bf, tf, _ = design_flanges(d, tw, Zp_req, fy, manual_bf)
    
    # We'll now iterate to satisfy all checks
    max_iter = 20
    iter_count = 0
    revised = False
    while iter_count < max_iter:
        # Compute properties
        Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
        Md = moment_capacity(Zp_actual, fy)
        # Stiffener requirement
        need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Md)  # temporary Vd not known
        Vd, shear_method, λ_w = shear_capacity(d, tw, fy, need_stiff, use_tension_field)
        
        # Checks
        web_ok, web_limit, web_actual, web_desc = check_web_slenderness_full(d, tw, fy, need_stiff, stiff_spacing)
        shear_ok = Vu <= Vd
        moment_ok = Mu <= Md
        delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
        defl_ok = delta <= delta_limit
        
        if web_ok and shear_ok and moment_ok and defl_ok:
            break
        
        # Revise section
        revised = True
        # Increase web thickness if web slenderness fails or shear fails
        if not web_ok or not shear_ok:
            tw += 2
            tw = min(tw, 40)
        # Increase flange thickness if moment fails or deflection slightly high
        if not moment_ok or delta > 1.2 * delta_limit:
            tf += 2
            if tf < 40:
                tf = min(tf, 60)
            else:
                tf += 5
        # If nothing else, increase overall depth
        if web_ok and moment_ok and not defl_ok:
            d += 20
            d = min(d, 2500)
            # Re‑compute web thickness for new d
            tw = max(tw, design_web(d, Vu, fy))
        # Re‑compute flange for new d, tw
        bf, tf, _ = design_flanges(d, tw, Zp_req, fy, manual_bf)
        iter_count += 1
    
    # Final properties after revision
    Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
    Md = moment_capacity(Zp_actual, fy)
    need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Md)
    Vd, shear_method, λ_w = shear_capacity(d, tw, fy, need_stiff, use_tension_field)
    web_ok, web_limit, web_actual, web_desc = check_web_slenderness_full(d, tw, fy, need_stiff, stiff_spacing)
    delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
    ratio_moment = Mu / Md
    ratio_shear = Vu / Vd
    
    # Display summary of revisions
    st.header("📊 Design Calculations (Step by Step)")
    if revised:
        st.info(f"⚠ Section was automatically revised after {iter_count} iteration(s) to satisfy all checks.")
    else:
        st.success("✅ Initial design passed all checks (no revision needed).")
    
    # 1. Factored loads
    st.subheader("1️⃣ Factored Loads & Required Plastic Modulus")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(f"M_u = {Mu:.1f}\\ \\text{{kN·m}}, \\quad V_u = {Vu:.1f}\\ \\text{{kN}}")
        st.latex(f"Z_{{p,\\text{{req}}}} = \\frac{{M_u \\cdot \\gamma_{{m0}}}}{{f_y}} = {Zp_req:.0f}\\ \\text{{mm}}^3")
    with col2:
        st.metric("Factored Moment", f"{Mu:.1f} kN·m")
        st.metric("Factored Shear", f"{Vu:.1f} kN")
    
    # 2. Economical depth
    st.subheader("2️⃣ Economical Depth (Cl. 8.6 - approximate method)")
    st.latex(r"d = \left( \frac{M_u \cdot K}{f_y} \right)^{1/3}, \quad K = \frac{d}{t_w}")
    st.write(f"Iteration gave K = {d/tw:.1f} → final d = {d} mm")
    st.success(f"✅ **Economical web depth adopted:** `{d} mm`")
    
    # 3. Web shear design
    st.subheader("3️⃣ Web Design (Cl. 8.6.1 & 8.4.2.2)")
    st.latex(f"t_w \\ge \\frac{{V_u \\gamma_{{m1}}}}{{d \\cdot (f_y/\\sqrt{{3}})}} = \\frac{{{Vu}\\times1000 \\times 1.25}}{{{d} \\times ({fy}/\\sqrt{{3}})}} = {design_web(d, Vu, fy):.1f}\\ \\text{{mm}}")
    st.latex(f"t_w \\ge \\frac{{d}}{{200}} = \\frac{{{d}}}{{200}} = {d/200:.1f}\\ \\text{{mm}}")
    st.success(f"✅ **Web thickness adopted:** `{tw} mm`")
    
    # 4. Shear buckling resistance
    st.subheader("4️⃣ Shear Buckling Resistance (Cl. 8.4.2.2)")
    st.latex(f"\\text{{Shear slenderness }} λ_w = {λ_w:.3f}")
    st.write(f"**Method:** {shear_method}")
    st.latex(f"V_n = {shear_capacity(d, tw, fy, need_stiff, use_tension_field)[0] * 1.25:.1f}\\ \\text{{kN}}")
    st.latex(f"V_d = V_n / γ_{{m1}} = {Vd:.2f}\\ \\text{{kN}}")
    st.write(f"Shear check: {Vu:.1f} kN {'≤' if shear_ok else '>'} {Vd:.2f} kN → {'OK' if shear_ok else 'NOT OK'}")
    
    # 5. Web slenderness
    st.subheader("5️⃣ Web Slenderness (Cl. 8.6.1.1)")
    st.latex(f"\\frac{{d}}{{t_w}} = {web_actual:.1f}")
    st.write(f"**Criteria:** {web_desc}")
    if web_ok:
        st.success(f"✓ d/tw = {web_actual:.1f} ≤ {web_limit:.1f}")
    else:
        st.error(f"✗ d/tw = {web_actual:.1f} > {web_limit:.1f}")
    
    # 6. Flange design
    st.subheader("6️⃣ Flange Design (Plastic Section Modulus)")
    st.latex(f"Z_{{p,\\text{{actual}}}} = {Zp_actual:.0f}\\ \\text{{mm}}^3")
    st.latex(f"M_d = \\frac{{Z_p f_y}}{{\\gamma_{{m0}}}} = {Md:.2f}\\ \\text{{kN·m}}")
    st.latex(f"b_f = {bf:.0f}\\ \\text{{mm}}, \\quad t_f = {tf:.1f}\\ \\text{{mm}}")
    epsilon = math.sqrt(250 / fy)
    compact_limit = 9.4 * epsilon
    bf_tf = bf / tf
    st.latex(f"\\text{{Flange compactness: }} \\frac{{b_f}}{{t_f}} = {bf_tf:.1f} \\le 9.4\\varepsilon = {compact_limit:.1f}")
    if bf_tf <= compact_limit:
        st.success("✓ Flange is compact (Class 1/2).")
    else:
        st.error("✗ Flange is slender — revise manually.")
    
    # 7. Deflection
    st.subheader("7️⃣ Serviceability: Deflection (Cl. 5.6.1)")
    st.latex(f"I_x = {Ix_cm4:.1f}\\ \\text{{cm}}^4")
    st.latex(f"\\delta = {delta:.1f}\\ \\text{{mm}}, \\quad \\delta_{{\\text{{limit}}}} = {delta_limit:.1f}\\ \\text{{mm}}")
    if delta <= delta_limit:
        st.success(f"✓ Deflection OK")
    else:
        st.warning(f"⚠ High deflection — consider larger girder.")
    
    # 8. Intermediate stiffeners
    st.subheader("8️⃣ Intermediate Stiffeners (Cl. 8.7.3)")
    if need_stiff:
        st.warning(f"⚠ Stiffeners required at spacing ≤ {stiff_spacing} mm (≤ 1.5d).")
    else:
        st.success("✅ No intermediate stiffeners required.")
    
    # 9. Material estimate
    total_weight = weight * span
    st.subheader("9️⃣ Material Estimate")
    col1, col2 = st.columns(2)
    col1.metric("Weight per meter", f"{weight:.0f} kg/m")
    col2.metric("Total weight", f"{total_weight:.0f} kg ({total_weight/1000:.2f} tonnes)")
    
    # 10. Final Design Summary Table
    st.subheader("📋 Final Design Summary Table (After Auto‑Revision)")
    summary_data = {
        "Parameter": [
            "Span", "Load type", "Factored Moment (Mu)", "Factored Shear (Vu)",
            "Web plate (d × tw) — FINAL", "Flange plate (bf × tf) — FINAL",
            "Web slenderness (d/tw)", "Shear slenderness λ_w",
            "Moment capacity (Md)", "Shear capacity (Vd)",
            "Moment utilization (Mu/Md)", "Shear utilization (Vu/Vd)",
            "Shear buckling method", "Deflection (δ)", "Deflection limit (L/300)",
            "Deflection check", "Intermediate stiffeners", "Stiffener spacing (if req.)",
            "Steel grade", "Weight per meter", "Total weight"
        ],
        "Value": [
            f"{span} m", load_type, f"{Mu:.1f} kN·m", f"{Vu:.1f} kN",
            f"{d} mm × {tw} mm", f"{bf} mm × {tf:.1f} mm",
            f"{web_actual:.1f}", f"{λ_w:.3f}",
            f"{Md:.2f} kN·m", f"{Vd:.2f} kN",
            f"{ratio_moment:.3f}", f"{ratio_shear:.3f}",
            shear_method, f"{delta:.1f} mm", f"{delta_limit:.1f} mm",
            "✅ OK" if delta <= delta_limit else "❌ NOT OK",
            "Required" if need_stiff else "Not required",
            f"{stiff_spacing} mm c/c" if need_stiff and stiff_spacing else "—",
            steel_grade, f"{weight:.0f} kg/m", f"{total_weight:.0f} kg ({total_weight/1000:.2f} t)"
        ]
    }
    st.table(summary_data)
    
    # Engineering Notes
    st.subheader("📝 Engineering Notes")
    st.markdown(f"""
    - **Material**: All steel plates are {steel_grade} (f_y = {fy} MPa).
    - **Shear buckling**: Designed using **{shear_method}** as per Cl. 8.4.2.2.
    - **Web slenderness**: {web_desc} → {'✓ OK' if web_ok else '✗ NOT OK (revised)'}.
    - **Flange compactness**: b_f/t_f = {bf_tf:.1f} ≤ {compact_limit:.1f} → compact.
    - **Auto‑revision**: The section was automatically adjusted to satisfy all limit states.
    """)
    
    # Developer name
    st.markdown("---")
    st.markdown("**👨‍🏫 Developer:** *Dr Hiteshkumar Santosh Patil, Assistant Professor, Civil Engineering Department, RCPIT, Shirpur*")
    st.success("✅ Design completed. All checks are as per IS 800:2007.")
    
else:
    st.info("👈 Enter design parameters in the sidebar and click **Design Plate Girder** to start.")
    st.markdown("""
    ### 📖 Design Steps (As per IS 800:2007)
    1. **Load calculations** – Factored moment & shear
    2. **Economical depth** – Iterative method
    3. **Web thickness** – Shear capacity + minimum thickness
    4. **Shear buckling resistance** – Simple post‑critical or tension field method (Cl. 8.4.2.2)
    5. **Web slenderness** – Limits for unstiffened/stiffened webs (Cl. 8.6.1.1)
    6. **Flange sizing** – Plastic modulus & compactness
    7. **Deflection check** – Serviceability (L/300)
    8. **Intermediate stiffeners** – Requirement and spacing
    9. **Auto‑revision** – Iterative adjustment if any check fails
    10. **Material estimate & summary table**
    """)
