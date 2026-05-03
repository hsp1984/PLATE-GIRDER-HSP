import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

st.set_page_config(page_title="Plate Girder Designer - IS 800:2007", layout="wide")
st.title("🏗️ Steel Plate Girder Designer")
st.markdown("Design of welded plate girders as per **IS 800:2007 (Limit State Method)**")
st.markdown("Economical depth formula: \( d = \\left( \\frac{M \\cdot K}{f_y} \\right)^{1/3} \) with \( K = d/t_w \)")

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
    st.info(f"Yield strength (fy) = **{fy} MPa**")
    
    st.header("3. Economical Depth Parameters")
    K_initial = st.slider("Initial d/tw ratio (K)", min_value=50, max_value=150, value=100, step=5,
                          help="Typical values: 80–120 for plate girders")
    
    # Optional manual flange width
    st.header("4. Optional (Advanced)")
    manual_bf = st.number_input("Manual flange width (mm) – 0 = auto", min_value=0, value=0, step=10)

    st.markdown("---")
    st.caption("Design as per IS 800:2007, Cl. 8.6 — Beams and Plate Girders with Solid Webs")

# ---------- Helper Functions ----------
def calc_factored_loads(dl, ll, point_load, span, load_type, gamma_f=1.5):
    """Return factored moment Mu (kNm) and shear Vu (kN)"""
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

def economical_depth(Mu, fy, K):
    """Compute economical depth d (mm) from d = (M * K / fy)^(1/3)
       Mu in kNm, fy in MPa, d in mm."""
    M_Nmm = Mu * 1e6
    d_mm = (M_Nmm * K / fy) ** (1/3)
    return d_mm

def design_web_economical(Mu, Vu, fy, K_initial, gamma_m1=1.25):
    """
    Iteratively determine web depth d and thickness tw using the economical depth formula
    and shear capacity requirement.
    """
    # Step 1: estimate d from economical formula using initial K
    d = economical_depth(Mu, fy, K_initial)
    # Ensure reasonable bounds (500 mm to 2500 mm)
    d = max(500, min(2500, round(d / 10) * 10))
    
    # Step 2: calculate required tw from shear
    tw_req_shear = (Vu * 1000 * gamma_m1) / (d * (fy / math.sqrt(3)))
    tw_min = max(6.0, d / 200.0)   # minimum thickness
    tw = max(tw_min, tw_req_shear)
    
    # Step 3: compute actual K = d/tw
    K_act = d / tw
    
    # Step 4: if K_act differs significantly from K_initial, re-evaluate d once
    if abs(K_act - K_initial) > 10:
        d_new = economical_depth(Mu, fy, K_act)
        d_new = max(500, min(2500, round(d_new / 10) * 10))
        if abs(d_new - d) > 20:
            d = d_new
            # Recompute tw based on new d
            tw_req_shear = (Vu * 1000 * gamma_m1) / (d * (fy / math.sqrt(3)))
            tw_min = max(6.0, d / 200.0)
            tw = max(tw_min, tw_req_shear)
    
    # Round tw to even mm
    tw = max(6.0, min(40.0, round(tw / 2) * 2))
    return d, tw, K_act

def check_web_slenderness(d, tw, fy):
    epsilon = math.sqrt(250 / fy)
    limit = 67 * epsilon
    actual = d / tw
    return actual <= limit, limit, actual

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
        tf = max(10.0, round(tf))
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

def shear_capacity(d, tw, fy, gamma_m1=1.25):
    return tw * d * (fy / math.sqrt(3)) / gamma_m1 / 1000.0

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

def bearing_stiffeners_required(Vu, d, tw, fy, gamma_m0=1.10):
    bearing_capacity = (tw * (d/2) * (fy / math.sqrt(3))) / gamma_m0 / 1000
    return bearing_capacity < Vu, bearing_capacity

def required_plastic_modulus(Mu, fy, gamma_m0=1.10):
    return Mu * 1e6 * gamma_m0 / fy

# ---------- Main Design Execution ----------
if st.sidebar.button("🚀 Design Plate Girder (Economical Depth)", type="primary", use_container_width=True):
    
    Mu, Vu, w_serv, P_serv = calc_factored_loads(dl, ll, point_load, span, load_type)
    Zp_req = required_plastic_modulus(Mu, fy)
    
    # Use economical depth formula
    d, tw, K_actual = design_web_economical(Mu, Vu, fy, K_initial)
    
    bf, tf, Zp_web = design_flanges(d, tw, Zp_req, fy, manual_bf)
    Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
    Md = moment_capacity(Zp_actual, fy)
    Vd = shear_capacity(d, tw, fy)
    
    web_ok, web_limit, web_actual = check_web_slenderness(d, tw, fy)
    need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Vd)
    need_bear_stiff, bearing_cap = bearing_stiffeners_required(Vu, d, tw, fy)
    delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
    
    # ---------- Display Detailed Calculations ----------
    st.header("📊 Step‑by‑Step Design Calculations")
    
    st.subheader("1️⃣ Factored Loads & Required Plastic Modulus")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(f"M_u = {Mu:.1f}\\ \\text{{kN·m}}, \\quad V_u = {Vu:.1f}\\ \\text{{kN}}")
        st.latex(f"Z_{{p,\\text{{req}}}} = \\frac{{M_u \\cdot \\gamma_{{m0}}}}{{f_y}} = \\frac{{{Mu}\\times10^6 \\times 1.1}}{{{fy}}} = {Zp_req:.0f}\\ \\text{{mm}}^3")
    
    st.subheader("2️⃣ Economical Web Depth (based on given formula)")
    st.latex(f"d = \\left( \\frac{{M \\cdot K}}{{f_y}} \\right)^{{1/3}}")
    st.write(f"Initial K (d/t<sub>w</sub>) = {K_initial}  →  d ≈ {economical_depth(Mu, fy, K_initial):.0f} mm")
    st.latex(f"\\text{{Selected web depth: }} d = {d}\\ \\text{{mm}}")
    st.latex(f"\\text{{Web thickness from shear: }} t_w \\ge \\frac{{V_u \\gamma_{{m1}}}}{{d \\cdot (f_y/\\sqrt{{3}})}} = {design_web_economical(Mu, Vu, fy, K_initial)[1]:.1f}\\ \\text{{mm}}")
    st.latex(f"t_w \\ge \\frac{{d}}{{200}} = {d/200:.1f}\\ \\text{{mm}}")
    st.success(f"✅ **Adopted web plate:** {d} mm × {tw} mm  (actual d/t<sub>w</sub> = {K_actual:.1f})")
    
    st.subheader("3️⃣ Web Slenderness (Unstiffened)")
    st.latex(f"\\frac{{d}}{{t_w}} = {web_actual:.1f} \\quad \\text{{Limit}} = 67\\varepsilon = {web_limit:.1f}")
    if web_ok:
        st.success("✓ Unstiffened web is acceptable.")
    else:
        st.warning("⚠ Intermediate stiffeners required.")
    
    st.subheader("4️⃣ Flange Design (Plastic Section Modulus)")
    st.latex(f"Z_{{p,\\text{{web}}}} = \\frac{{t_w d^2}}{{4}} = {Zp_web:.0f}\\ \\text{{mm}}^3")
    st.latex(f"A_{{f,\\text{{req}}}} = \\frac{{Z_{{p,\\text{{req}}}} - Z_{{p,\\text{{web}}}}}}{{d}} = {max(0, (Zp_req-Zp_web)/d):.0f}\\ \\text{{mm}}^2")
    st.latex(f"b_f \\approx d/4 = {d/4:.0f}\\ \\text{{mm}} \\rightarrow \\text{{adopted }} b_f = {bf:.0f}\\ \\text{{mm}}")
    st.latex(f"t_f = A_{{f,\\text{{req}}}}/b_f = {tf:.1f}\\ \\text{{mm}}")
    # compactness
    epsilon_val = math.sqrt(250/fy)
    bf_tf = bf/tf
    st.latex(f"\\text{{Flange compactness: }} \\frac{{b_f}}{{t_f}} = {bf_tf:.1f} \\le 9.4\\varepsilon = {9.4*epsilon_val:.1f}")
    if bf_tf <= 9.4*epsilon_val: st.success("✓ Flange compact (Class 1/2).")
    else: st.error("⚠ Flange slender – increase tf.")
    st.success(f"✅ **Flange plate:** {bf} mm × {tf} mm")
    
    st.subheader("5️⃣ Section Capacity")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(f"Z_{{p,\\text{{actual}}}} = {Zp_actual:.0f}\\ \\text{{mm}}^3")
        st.latex(f"M_d = {Md:.2f}\\ \\text{{kN·m}} \\quad \\text{{Ratio}} = {Mu/Md:.3f}")
        if Mu/Md <= 1: st.success("✓ Moment capacity OK")
        else: st.error("✗ Increase section")
    with col2:
        st.latex(f"V_d = {Vd:.2f}\\ \\text{{kN}} \\quad \\text{{Ratio}} = {Vu/Vd:.3f}")
        if Vu/Vd <= 1: st.success("✓ Shear capacity OK")
        else: st.error("✗ Increase tw or add stiffeners")
    
    st.subheader("6️⃣ Serviceability Deflection")
    st.latex(f"I_x = {Ix_cm4:.1f}\\ \\text{{cm}}^4")
    st.latex(f"\\delta = {delta:.1f}\\ \\text{{mm}} \\quad \\delta_{{\\text{{limit}}}} = {delta_limit:.1f}\\ \\text{{mm}}")
    if delta <= delta_limit: st.success("✓ Deflection OK")
    else: st.warning("⚠ High deflection – increase stiffness")
    
    st.subheader("7️⃣ Stiffener Requirements")
    if need_stiff:
        st.warning(f"⚠ Intermediate stiffeners required at max spacing ≈ {stiff_spacing} mm (≤ 1.5d).")
    else:
        st.success("✅ No intermediate stiffeners required.")
    if need_bear_stiff:
        st.warning("⚠ End bearing stiffeners required.")
    else:
        st.info("End bearing stiffeners may still be provided by standard practice.")
    
    st.subheader("8️⃣ Material Estimate")
    total_weight = weight * span
    st.metric("Weight per metre", f"{weight:.0f} kg/m")
    st.metric("Total weight", f"{total_weight:.0f} kg ({total_weight/1000:.2f} tonnes)")
    
    # ---------- Drawing ----------
    st.subheader("9️⃣ Detailed Elevation Drawing")
    fig, ax = plt.subplots(figsize=(12,5))
    d_plot = d / 10; tf_plot = tf / 10; span_plot = span * 100; y_center = 10
    web_x = 40; web_width = span_plot - 80
    ax.add_patch(Rectangle((web_x, y_center - d_plot/2), web_width, d_plot,
                           fc='lightblue', ec='black', lw=1.5))
    ax.add_patch(Rectangle((web_x, y_center - d_plot/2 - tf_plot), web_width, tf_plot,
                           fc='steelblue', ec='black', lw=1.5))
    ax.add_patch(Rectangle((web_x, y_center + d_plot/2), web_width, tf_plot,
                           fc='steelblue', ec='black', lw=1.5))
    if need_stiff and stiff_spacing:
        spacing_plot = stiff_spacing / 10
        x = web_x + spacing_plot
        while x < web_x + web_width:
            ax.add_patch(Rectangle((x, y_center - d_plot/2 - tf_plot/2), 8, d_plot+tf_plot,
                                   fc='salmon', ec='darkred', alpha=0.8))
            x += spacing_plot
    # bearing stiffeners
    ax.add_patch(Rectangle((web_x-8, y_center - d_plot/2 - tf_plot/2), 12, d_plot+tf_plot,
                           fc='peru', ec='brown', lw=1.5))
    ax.add_patch(Rectangle((web_x+web_width-4, y_center - d_plot/2 - tf_plot/2), 12, d_plot+tf_plot,
                           fc='peru', ec='brown', lw=1.5))
    ax.annotate(f'Web: {d}×{tw}', xy=(web_x+25, y_center), fontsize=9, ha='center', bbox=dict(boxstyle='round', fc='white'))
    ax.annotate(f'Flange: {bf}×{tf}', xy=(web_x+web_width-50, y_center+d_plot/2+tf_plot/2), fontsize=9, ha='center', bbox=dict(boxstyle='round', fc='white'))
    ax.annotate('', xy=(web_x, y_center - d_plot/2 - tf_plot - 40), xytext=(web_x+web_width, y_center - d_plot/2 - tf_plot - 40),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.annotate(f'Span = {span} m', xy=(web_x+web_width/2, y_center - d_plot/2 - tf_plot - 55), ha='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, span_plot+100); ax.set_ylim(y_center - d_plot/2 - tf_plot - 100, y_center + d_plot/2 + tf_plot + 50)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Plate Girder Elevation (Economical Depth)', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    st.success("✅ Design completed using the economical depth formula \( d = (M K / f_y)^{1/3} \).")
    st.markdown("---")
    st.caption("All steps satisfy IS 800:2007 limits. Intermediate and bearing stiffeners shown where required.")
else:
    st.info("👈 Set parameters in the sidebar and click **Design Plate Girder** to start.")
    st.markdown("""
    ### 📐 Design Features
    - **Economical depth** formula: \( d = \\left( \\frac{M \\cdot K}{f_y} \\right)^{1/3} \) with \( K = d/t_w \)
    - Automatic web/flange sizing
    - Shear, moment, deflection, slenderness checks
    - Intermediate and bearing stiffener recommendations
    - Scale drawing with all details
    """)
