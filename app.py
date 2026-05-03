import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math

st.set_page_config(page_title="Plate Girder Designer - IS 800:2007", layout="wide")
st.title("🏗️ Steel Plate Girder Designer (Economical Depth + Weld Design)")
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
    fu = 410 if fy == 250 else 550  # approximate ultimate strength for weld design
    st.info(f"Yield strength (fy) = **{fy} MPa**  |  Ultimate strength (fu) = **{fu} MPa**")
    
    st.header("3. Optional (Advanced)")
    manual_bf = st.number_input("Manual flange width (mm) – 0 = auto", min_value=0, value=0, step=10)
    st.markdown("---")
    st.caption("Design as per IS 800:2007, Cl. 8.6 — Beams and Plate Girders")

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

def economical_depth(Mu_kNm, fy, K=100):
    """Iterative economical depth: d = (M * K / fy)^(1/3)  (M in Nmm, d in mm)"""
    Mu_Nmm = Mu_kNm * 1e6
    d = (Mu_Nmm * K / fy) ** (1/3)
    # round to nearest 10 mm
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

def weld_design(Vu, Ix_cm4, bf, tf, fu):
    """
    Design flange-to-web fillet weld.
    Horizontal shear force per unit length (N/mm): q = (Vu * A_f * y_bar) / I
    Where A_f = bf * tf, y_bar = (d + tf)/2, I = Ix_cm4 * 1e4 (mm⁴)
    Required weld leg size (mm) for double continuous fillet weld.
    """
    # Convert Ix to mm⁴
    Ix_mm4 = Ix_cm4 * 1e4
    # First moment of area of one flange about neutral axis
    y_bar = (d + tf) / 2
    Q = bf * tf * y_bar   # mm³
    # Shear flow (N/mm) at the interface
    shear_flow = Vu * 1000 * Q / Ix_mm4   # N/mm
    
    # Design strength of fillet weld (Cl. 10.5.4.1)
    # For shop welding, use partial safety factor γmw = 1.25 (Table 5)
    gamma_mw = 1.25
    # Effective throat thickness = 0.7 * leg_size
    # Weld strength per mm run = (fu/√3) / γmw * (0.7 * s)
    # Double continuous weld: total strength = 2 * single weld strength
    # Equate to shear_flow to find required leg size s
    # 2 * (0.7 * s) * (fu/√3) / γmw = shear_flow
    # Therefore s = (shear_flow * γmw * √3) / (2 * 0.7 * fu)
    s_req = (shear_flow * gamma_mw * math.sqrt(3)) / (2 * 0.7 * fu)
    # Minimum weld size (Cl. 10.5.2.3) based on thickness of thicker part (flange)
    min_weld = max(3, math.ceil(tf / 2) if tf < 20 else 6)
    s_design = max(min_weld, math.ceil(s_req))
    return s_design, shear_flow

# ---------- Main Design Execution ----------
if st.sidebar.button("🚀 Design Plate Girder", type="primary", use_container_width=True):
    
    Mu, Vu, w_serv, P_serv = calc_factored_loads(dl, ll, point_load, span, load_type)
    
    st.header("📊 Design Calculations (Step by Step)")
    
    # 1. Economical depth (iterative)
    # Start with assumed K = d/tw = 100 (typical)
    K_initial = 100
    d = economical_depth(Mu, fy, K_initial)
    tw = design_web(d, Vu, fy)
    # Recalculate d using actual K = d/tw
    K_actual = d / tw
    d = economical_depth(Mu, fy, K_actual)
    # Round to nearest 10 mm
    d = max(400, min(3000, round(d / 10) * 10))
    tw = design_web(d, Vu, fy)   # final web thickness
    
    st.subheader("1️⃣ Economical Depth (Cl. 8.6 - approximate method)")
    st.latex(r"d = \left( \frac{M_u \cdot K}{f_y} \right)^{1/3}, \quad K = \frac{d}{t_w}")
    st.write(f"After iteration: K ≈ {d/tw:.1f}")
    st.latex(f"d = \\left( \\frac{{{Mu}\\times10^6 \\times {d/tw:.1f}}}{{{fy}}} \\right)^{{1/3}} = {d}\\ \\text{{mm}}")
    st.success(f"✅ **Economical web depth:** `{d} mm`")

    # 2. Factored loads and required Zp
    Zp_req = required_plastic_modulus(Mu, fy)
    st.subheader("2️⃣ Factored Loads & Required Plastic Modulus")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(f"M_u = {Mu:.1f}\\ \\text{{kN·m}}, \\quad V_u = {Vu:.1f}\\ \\text{{kN}}")
        st.latex(f"Z_{{p,\\text{{req}}}} = \\frac{{M_u \\cdot \\gamma_{{m0}}}}{{f_y}} = {Zp_req:.0f}\\ \\text{{mm}}^3")
    with col2:
        st.metric("Factored Moment", f"{Mu:.1f} kN·m")
        st.metric("Factored Shear", f"{Vu:.1f} kN")

    # 3. Web design
    st.subheader("3️⃣ Web Design (Cl. 8.6.1 & 8.4.2.2)")
    st.latex(f"t_w \\ge \\frac{{V_u \\gamma_{{m1}}}}{{d \\cdot (f_y/\\sqrt{{3}})}} = \\frac{{{Vu}\\times1000 \\times 1.25}}{{{d} \\times ({fy}/\\sqrt{{3}})}} = {design_web(d, Vu, fy):.1f}\\ \\text{{mm}}")
    st.latex(f"t_w \\ge \\frac{{d}}{{200}} = \\frac{{{d}}}{{200}} = {d/200:.1f}\\ \\text{{mm}}")
    st.success(f"✅ **Web thickness adopted:** `{tw} mm`")

    # 4. Web slenderness check
    web_ok, web_limit, web_actual = check_web_slenderness(d, tw, fy)
    st.subheader("4️⃣ Web Slenderness (Unstiffened)")
    st.latex(f"\\frac{{d}}{{t_w}} = {web_actual:.1f} \\le 67\\varepsilon = 67\\sqrt{{\\frac{{250}}{{{fy}}}}} = {web_limit:.1f}")
    if web_ok:
        st.success("✓ Unstiffened web is acceptable.")
    else:
        st.warning("⚠ Intermediate stiffeners required.")

    # 5. Flange design
    bf, tf, Zp_web = design_flanges(d, tw, Zp_req, fy, manual_bf)
    st.subheader("5️⃣ Flange Design (Plastic Section Modulus)")
    st.latex(f"Z_{{p,\\text{{web}}}} = \\frac{{t_w d^2}}{{4}} = {Zp_web:.0f}\\ \\text{{mm}}^3")
    Af_req = max(0, (Zp_req - Zp_web) / d)
    st.latex(f"A_{{f,\\text{{req}}}} = \\frac{{{Zp_req:.0f} - {Zp_web:.0f}}}{{{d}}} = {Af_req:.0f}\\ \\text{{mm}}^2")
    st.latex(f"b_f \\text{{ (recommended)}} \\approx d/4 = {d/4:.0f}\\ \\text{{mm}} \\rightarrow \\text{{Adopted }} b_f = {bf:.0f}\\ \\text{{mm}}")
    st.latex(f"t_f = \\frac{{A_{{f,\\text{{req}}}}}}{{b_f}} = {tf:.1f}\\ \\text{{mm}}")
    # Flange compactness
    epsilon = math.sqrt(250 / fy)
    compact_limit = 9.4 * epsilon
    bf_tf = bf / tf
    st.latex(f"\\text{{Flange compactness: }} \\frac{{b_f}}{{t_f}} = {bf_tf:.1f} \\le 9.4\\varepsilon = {compact_limit:.1f}")
    if bf_tf <= compact_limit:
        st.success("✓ Flange is compact (Class 1/2).")
    else:
        st.error("⚠ Flange is slender — increase tf or reduce bf.")

    # 6. Section properties & capacity
    Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
    Md = moment_capacity(Zp_actual, fy)
    Vd = shear_capacity(d, tw, fy)
    st.subheader("6️⃣ Section Capacity (Cl. 8.2.1.2 & 8.4.2.2)")
    st.latex(f"Z_{{p,\\text{{actual}}}} = {Zp_actual:.0f}\\ \\text{{mm}}^3")
    st.latex(f"M_d = \\frac{{Z_p f_y}}{{\\gamma_{{m0}}}} = {Md:.2f}\\ \\text{{kN·m}}")
    st.latex(f"V_d = \\frac{{t_w d (f_y/\\sqrt{{3}})}}{{\\gamma_{{m1}}}} = {Vd:.2f}\\ \\text{{kN}}")
    ratio_moment = Mu / Md
    ratio_shear = Vu / Vd
    col1, col2 = st.columns(2)
    col1.metric("Moment ratio", f"{ratio_moment:.3f}", help="≤ 1.0 OK")
    col2.metric("Shear ratio", f"{ratio_shear:.3f}", help="≤ 1.0 OK")
    if ratio_moment <= 1.0 and ratio_shear <= 1.0:
        st.success("✓ Moment and shear capacities are adequate.")
    else:
        st.error("✗ Capacity insufficient — revise section.")

    # 7. Deflection
    delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
    st.subheader("7️⃣ Serviceability: Deflection (Cl. 5.6.1)")
    st.latex(f"I_x = {Ix_cm4:.1f}\\ \\text{{cm}}^4")
    if load_type == "Uniformly Distributed Load (UDL)":
        st.latex(f"\\delta = \\frac{{5 w L^4}}{{384 E I}} = \\frac{{5 \\times {w_serv:.2f} \\times ({span*1000})^4}}{{384 \\times 2\\times10^5 \\times {Ix_cm4:.1f}\\times10^4}} = {delta:.1f}\\ \\text{{mm}}")
    else:
        st.latex(f"\\delta = \\frac{{P L^3}}{{48 E I}} = \\frac{{{P_serv:.2f}\\times1000 \\times ({span*1000})^3}}{{48 \\times 2\\times10^5 \\times {Ix_cm4:.1f}\\times10^4}} = {delta:.1f}\\ \\text{{mm}}")
    st.latex(f"\\delta_{{\\text{{limit}}}} = L/300 = {delta_limit:.1f}\\ \\text{{mm}}")
    if delta <= delta_limit:
        st.success(f"✓ Deflection OK: {delta:.1f} mm ≤ {delta_limit:.1f} mm")
    else:
        st.warning(f"⚠ High deflection — increase girder stiffness.")

    # 8. Stiffener requirements
    need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Vd)
    st.subheader("8️⃣ Intermediate Stiffeners (Cl. 8.7.3)")
    if need_stiff:
        st.warning(f"⚠ Stiffeners required at spacing ≤ {stiff_spacing} mm (≤ 1.5d).")
    else:
        st.success("✅ No intermediate stiffeners required.")

    # 9. Weld design
    s_weld, shear_flow = weld_design(Vu, Ix_cm4, bf, tf, fu)
    st.subheader("9️⃣ Flange-to-Web Fillet Weld (Cl. 10.5.4)")
    st.latex(f"\\text{{Shear flow at interface: }} q = \\frac{{V_u \\cdot Q}}{{I}} = \\frac{{{Vu}\\times1000 \\times ({bf}\\times{tf}\\times{(d+tf)/2})}}{{{Ix_cm4:.1f}\\times10^4}} = {shear_flow:.1f}\\ \\text{{N/mm}}")
    st.latex(f"\\text{{Required weld leg size (double continuous) }} s \\ge \\frac{{q \\cdot \\gamma_{{mw}} \\cdot \\sqrt{{3}}}}{{2 \\cdot 0.7 \\cdot f_u}} = \\frac{{{shear_flow:.1f} \\times 1.25 \\times \\sqrt{{3}}}}{{2 \\times 0.7 \\times {fu}}} = {s_weld-0.1:.1f}\\ \\text{{mm}}")
    st.success(f"✅ **Provide double continuous fillet weld of leg size `{s_weld} mm`**")

    # 10. Material estimate
    total_weight = weight * span
    st.subheader("🔟 Material Estimate")
    col1, col2 = st.columns(2)
    col1.metric("Weight per meter", f"{weight:.0f} kg/m")
    col2.metric("Total weight", f"{total_weight:.0f} kg ({total_weight/1000:.2f} tonnes)")

    # 11. Elevation drawing with weld indication
    st.subheader("📐 Detailed Elevation Drawing")
    fig, ax = plt.subplots(figsize=(12, 5))
    d_plot = d / 10; tf_plot = tf / 10; span_plot = span * 100; y_center = 10
    web_x = 40; web_width = span_plot - 80
    # Web
    ax.add_patch(Rectangle((web_x, y_center - d_plot/2), web_width, d_plot,
                           fc='lightblue', ec='black', lw=1.5))
    # Top flange
    ax.add_patch(Rectangle((web_x, y_center - d_plot/2 - tf_plot), web_width, tf_plot,
                           fc='steelblue', ec='black', lw=1.5))
    # Bottom flange
    ax.add_patch(Rectangle((web_x, y_center + d_plot/2), web_width, tf_plot,
                           fc='steelblue', ec='black', lw=1.5))
    # Weld indication (dashed lines on both sides)
    weld_offset = 2  # mm in plot units (0.2 cm)
    for side in [-1, 1]:  # both sides of web
        weld_y = y_center + side * (d_plot/2)
        ax.plot([web_x, web_x+web_width], [weld_y, weld_y], 'r--', lw=2, label='Weld' if side==-1 else "")
        ax.plot([web_x, web_x+web_width], [weld_y + side*tf_plot, weld_y + side*tf_plot], 'r--', lw=2)
    # Stiffeners if needed
    if need_stiff and stiff_spacing:
        spacing_plot = stiff_spacing / 10
        stiff_width = 8
        x_start = web_x + spacing_plot
        while x_start < web_x + web_width:
            ax.add_patch(Rectangle((x_start, y_center - d_plot/2 - tf_plot/2),
                                   stiff_width, d_plot + tf_plot,
                                   fc='salmon', ec='darkred', alpha=0.7))
            x_start += spacing_plot
    # Bearing stiffeners (end)
    end_stiff_width = 12
    ax.add_patch(Rectangle((web_x - end_stiff_width/2, y_center - d_plot/2 - tf_plot/2),
                           end_stiff_width, d_plot + tf_plot, fc='peru', ec='brown'))
    ax.add_patch(Rectangle((web_x + web_width - end_stiff_width/2, y_center - d_plot/2 - tf_plot/2),
                           end_stiff_width, d_plot + tf_plot, fc='peru', ec='brown'))
    # Labels
    ax.annotate(f'Web: {d}×{tw}', xy=(web_x+25, y_center), ha='center', bbox=dict(boxstyle='round', fc='white'))
    ax.annotate(f'Flange: {bf}×{tf}', xy=(web_x+web_width-50, y_center+d_plot/2+tf_plot/2), ha='center')
    ax.annotate(f'Weld: {s_weld} mm (double)', xy=(web_x+web_width-100, y_center+d_plot/2+tf_plot+5), ha='center', fontsize=8, color='red')
    # Span line
    ax.annotate('', xy=(web_x, y_center - d_plot/2 - tf_plot - 40),
                xytext=(web_x+web_width, y_center - d_plot/2 - tf_plot - 40),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.annotate(f'Span = {span} m', xy=(web_x+web_width/2, y_center - d_plot/2 - tf_plot - 55),
                ha='center', fontsize=11, fontweight='bold')
    # Supports
    sup_w, sup_h = 30, 20
    ax.add_patch(Rectangle((web_x-18, y_center - d_plot/2 - tf_plot - sup_h), sup_w, sup_h, fc='gray', ec='black'))
    ax.add_patch(Rectangle((web_x+web_width-12, y_center - d_plot/2 - tf_plot - sup_h), sup_w, sup_h, fc='gray', ec='black'))
    ax.set_xlim(0, span_plot+120)
    ax.set_ylim(y_center - d_plot/2 - tf_plot - 100, y_center + d_plot/2 + tf_plot + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Plate Girder Elevation (Weld shown by red dashed lines)', fontsize=14)
    st.pyplot(fig)
    
    st.success("✅ Design completed. All checks are as per IS 800:2007.")
    st.markdown("---")
    st.caption("Note: Economical depth formula is an initial approximation. Weld design follows Cl. 10.5.4. Double continuous fillet weld assumed on both sides of web.")
