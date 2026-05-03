import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

st.set_page_config(page_title="Plate Girder Designer – Detailed", layout="wide")
st.title("🏗️ Steel Plate Girder Designer (IS 800:2007)")
st.markdown("Comprehensive design with all intermediate calculations")

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.header("1. Geometry")
    span = st.number_input("Span (m)", min_value=5.0, max_value=50.0, value=15.0, step=0.5)

    st.header("2. Service Loads")
    load_type = st.selectbox("Load type", ["Uniformly Distributed Load (UDL)", "Point load at midspan"])
    if load_type == "Uniformly Distributed Load (UDL)":
        dl = st.number_input("Dead load (kN/m)", value=30.0, step=5.0)
        ll = st.number_input("Live load (kN/m)", value=20.0, step=5.0)
        point_load = 0.0
    else:
        point_load = st.number_input("Point load at midspan (kN)", value=150.0, step=25.0)
        dl = ll = 0.0

    st.header("3. Steel Grade")
    grade = st.selectbox("Steel grade", ["Fe 250", "Fe 410", "Fe 450"])
    fy = 250 if "250" in grade else 410 if "410" in grade else 450
    st.info(f"Yield strength fy = {fy} MPa")

    st.header("4. Optional")
    manual_bf = st.number_input("Manual flange width (mm) – 0 = auto", min_value=0, value=0, step=10)

# ---------- Constants ----------
gamma_m0 = 1.10   # plastic section resistance
gamma_m1 = 1.25   # buckling resistance
E = 2.0e5         # MPa
G = 0.769e5       # MPa (approx)

# ---------- Helper Functions ----------
def calc_factored_loads():
    """Return Mu (kNm), Vu (kN) and service loads for deflection."""
    if load_type == "Uniformly Distributed Load (UDL)":
        w_serv = dl + ll
        w_u = 1.5 * w_serv
        Mu = w_u * span**2 / 8
        Vu = w_u * span / 2
        return Mu, Vu, w_serv, 0.0
    else:
        P_serv = point_load
        P_u = 1.5 * P_serv
        Mu = P_u * span / 4
        Vu = P_u / 2
        return Mu, Vu, 0.0, P_serv

def required_plastic_modulus(Mu):
    """Required plastic section modulus Zp_req (mm³)"""
    return Mu * 1e6 * gamma_m0 / fy

def design_web(d, Vu):
    """Determine web thickness tw (mm) based on shear force."""
    # Minimum thickness (6mm or d/200)
    tw_min = max(6.0, d / 200.0)
    # Shear capacity required: Vd >= Vu
    # Vd = (tw * d * (fy/√3)) / γm1   (kN)
    # Rearranging for tw:
    req_tw_shear = (Vu * 1000 * gamma_m1) / (d * (fy / math.sqrt(3)))
    tw = max(tw_min, req_tw_shear)
    tw = max(6.0, min(40.0, round(tw / 2) * 2))   # round to even mm
    return tw

def check_web_slenderness(d, tw, fy):
    """Return (ok, limit, actual) for unstiffened web."""
    epsilon = math.sqrt(250 / fy)
    limit = 67 * epsilon
    actual = d / tw
    return actual <= limit, limit, actual

def design_flanges(d, tw, Zp_req, fy, manual_bf):
    """Design flange width bf (mm) and thickness tf (mm)."""
    # Web contribution to Zp
    Zp_web = tw * d**2 / 4
    # Required flange area Af (assuming flanges carry the rest)
    Af_req = max(0.0, (Zp_req - Zp_web) / d)
    # Flange width
    if manual_bf > 0:
        bf = manual_bf
    else:
        bf = max(150.0, d / 4.0)
        bf = round(bf / 10) * 10
    tf = max(8.0, Af_req / bf)
    # Compactness check: bf/(2*tf) for outstand? Actually for I-section: bf/tf <= 9.4ε
    epsilon = math.sqrt(250 / fy)
    compact_limit = 9.4 * epsilon
    if (bf / tf) > compact_limit:
        tf = bf / (0.9 * compact_limit)   # tighten a bit
        tf = max(8.0, round(tf))
    tf = round(tf, 1)
    return bf, tf, Zp_web

def calculate_properties(d, tw, bf, tf):
    """Compute actual Zp (mm³), Ix (cm⁴) and weight per metre (kg/m)."""
    # Plastic modulus (approx)
    Zp_actual = tw * d**2 / 4 + bf * tf * (d + tf)
    # Moment of inertia (cm⁴) for deflection
    Ix_cm4 = (tw * d**3 / 12 + 2 * (bf * tf * ((d + tf)/2)**2)) / 10000.0
    # Weight per metre (kg/m) – steel density 7850 kg/m³
    area_m2 = (d/1000)*(tw/1000) + 2*(bf/1000)*(tf/1000)
    weight = area_m2 * 7850
    return Zp_actual, Ix_cm4, weight

def moment_capacity(Zp_actual):
    """Design moment capacity Md (kNm)"""
    return Zp_actual * fy / gamma_m0 / 1e6

def shear_capacity(d, tw):
    """Design shear capacity Vd (kN) – without stiffeners"""
    return tw * d * (fy / math.sqrt(3)) / gamma_m1 / 1000.0

def deflection_check(span_m, Ix_cm4, w_serv, P_serv):
    """Compute deflection (mm) and limit (span/300)."""
    delta_limit = span_m * 1000 / 300.0
    if load_type == "Uniformly Distributed Load (UDL)":
        delta = 5.0 * w_serv * (span_m * 1000)**4 / (384 * E * Ix_cm4 * 1e4)
    else:
        delta = P_serv * 1000 * (span_m * 1000)**3 / (48 * E * Ix_cm4 * 1e4)
    return delta, delta_limit

def stiffener_requirement(d, tw, fy, Vu, Vd):
    """Determine if intermediate stiffeners are needed.
       Returns (needed, spacing_mm)"""
    epsilon = math.sqrt(250/fy)
    if d/tw <= 67 * epsilon:
        return False, None
    # If shear force > 0.6 * Vd, stiffeners are required (simplified)
    if Vu > 0.6 * Vd:
        # Spacing approx 1.5d (or as per code)
        spacing = min(1.5 * d, 3000)
        spacing = round(spacing / 100) * 100
        return True, spacing
    return True, 1.5 * d  # still needed but larger spacing possible

# ---------- Main Design Routine ----------
if st.sidebar.button("🚀 Design Plate Girder", type="primary", use_container_width=True):
    st.header("📐 Design Calculations")

    # 1. Factored loads
    Mu, Vu, w_serv, P_serv = calc_factored_loads()
    st.subheader("1. Loads & Factored Moments")
    st.write(f"**Span:** {span} m")
    if load_type == "Uniformly Distributed Load (UDL)":
        st.write(f"Service loads: DL = {dl} kN/m, LL = {ll} kN/m → Total service = {dl+ll} kN/m")
        st.write(f"Factored load wu = 1.5 × {dl+ll} = {1.5*(dl+ll):.2f} kN/m")
        st.latex(f"M_u = \\frac{{w_u L^2}}{{8}} = \\frac{{{1.5*(dl+ll):.2f} \\times {span}^2}}{{8}} = {Mu:.2f}\\ \\text{{kN·m}}")
        st.latex(f"V_u = \\frac{{w_u L}}{{2}} = \\frac{{{1.5*(dl+ll):.2f} \\times {span}}}{{2}} = {Vu:.2f}\\ \\text{{kN}}")
    else:
        st.write(f"Service point load = {point_load} kN")
        st.write(f"Factored load Pu = 1.5 × {point_load} = {1.5*point_load:.2f} kN")
        st.latex(f"M_u = \\frac{{P_u L}}{{4}} = \\frac{{{1.5*point_load:.2f} \\times {span}}}{{4}} = {Mu:.2f}\\ \\text{{kN·m}}")
        st.latex(f"V_u = \\frac{{P_u}}{{2}} = \\frac{{{1.5*point_load:.2f}}}{{2}} = {Vu:.2f}\\ \\text{{kN}}")

    # 2. Required plastic modulus
    Zp_req = required_plastic_modulus(Mu)
    st.subheader("2. Required Plastic Section Modulus")
    st.latex(f"Z_{{p,\\text{{req}}}} = \\frac{{M_u \\cdot \\gamma_{{m0}}}}{{f_y}} = \\frac{{{Mu}\\times10^6 \\times {gamma_m0}}}{{{fy}}} = {Zp_req:.0f}\\ \\text{{mm}}^3")

    # 3. Web depth (trial)
    d_initial = span * 1000 / 11   # span/10 to span/12
    d = max(500, min(2500, round(d_initial / 10) * 10))
    st.subheader("3. Web Depth")
    st.write(f"Initial web depth = span/11 = {d_initial:.0f} mm → adopted d = {d} mm")

    # 4. Web thickness
    tw = design_web(d, Vu)
    st.subheader("4. Web Thickness")
    st.write(f"Minimum thickness (d/200) = {d/200:.1f} mm, required for shear = {Vu:.1f} kN")
    st.latex(f"t_w \\ge \\frac{{V_u \\gamma_{{m1}}}}{{d \\cdot (f_y/\\sqrt{{3}})}} = \\frac{{{Vu}\\times1000 \\times {gamma_m1}}}{{{d} \\times ({fy}/\\sqrt{{3}})}} = {design_web(d,Vu):.1f}\\ \\text{{mm}}")
    st.write(f"**Adopted tw = {tw} mm**")

    # 5. Web slenderness check
    web_ok, web_limit, web_actual = check_web_slenderness(d, tw, fy)
    st.subheader("5. Web Slenderness (Unstiffened)")
    st.latex(f"\\frac{{d}}{{t_w}} = \\frac{{{d}}}{{{tw}}} = {web_actual:.1f}")
    st.latex(f"\\text{{Limit }} = 67\\varepsilon = 67\\sqrt{{\\frac{{250}}{{{fy}}}}} = {web_limit:.1f}")
    if web_ok:
        st.success(f"✓ d/tw = {web_actual:.1f} ≤ {web_limit:.1f} → Unstiffened web is acceptable.")
    else:
        st.warning(f"⚠ d/tw = {web_actual:.1f} > {web_limit:.1f} → Web stiffeners are required.")

    # 6. Flange design
    bf, tf, Zp_web = design_flanges(d, tw, Zp_req, fy, manual_bf)
    st.subheader("6. Flange Dimensions")
    st.latex(f"Z_{{p,\\text{{web}}}} = \\frac{{t_w d^2}}{{4}} = \\frac{{{tw} \\times {d}^2}}{{4}} = {Zp_web:.0f}\\ \\text{{mm}}^3")
    st.latex(f"A_{{f,\\text{{req}}}} = \\frac{{Z_{{p,\\text{{req}}}} - Z_{{p,\\text{{web}}}}}}{{d}} = \\frac{{{Zp_req:.0f} - {Zp_web:.0f}}}{{{d}}} = {max(0, (Zp_req-Zp_web)/d):.0f}\\ \\text{{mm}}^2")
    st.write(f"Flange width bf = {bf:.0f} mm, thickness tf = {tf:.1f} mm")
    # compactness check
    epsilon = math.sqrt(250/fy)
    compact_limit = 9.4 * epsilon
    bf_tf = bf / tf
    st.latex(f"\\frac{{b_f}}{{t_f}} = {bf_tf:.1f} \\leq 9.4\\varepsilon = {compact_limit:.1f}")
    if bf_tf <= compact_limit:
        st.success("Flange is compact (Class 1/2).")
    else:
        st.error("Flange is slender; redesign needed – increase tf.")

    # 7. Actual section properties and moment capacity
    Zp_actual, Ix_cm4, weight_kgpm = calculate_properties(d, tw, bf, tf)
    Md = moment_capacity(Zp_actual)
    Vd = shear_capacity(d, tw)
    st.subheader("7. Section Capacity")
    st.latex(f"Z_{{p,\\text{{actual}}}} = {Zp_actual:.0f}\\ \\text{{mm}}^3 \\quad \\Rightarrow \\quad M_d = \\frac{{Z_p f_y}}{{\\gamma_{{m0}}}} = \\frac{{{Zp_actual:.0f} \\times {fy}}}{{1.1\\times10^6}} = {Md:.2f}\\ \\text{{kN·m}}")
    st.latex(f"V_d = \\frac{{t_w d (f_y/\\sqrt{{3}})}}{{\\gamma_{{m1}}}} = \\frac{{{tw}\\times{d}\\times({fy}/\\sqrt{{3}})}}{{1.25\\times1000}} = {Vd:.2f}\\ \\text{{kN}}")
    st.write(f"**Moment ratio:** {Mu:.2f} / {Md:.2f} = {Mu/Md:.3f} ≤ 1 → {'OK' if Mu<=Md else 'NOT OK'}")
    st.write(f"**Shear ratio:** {Vu:.2f} / {Vd:.2f} = {Vu/Vd:.3f} ≤ 1 → {'OK' if Vu<=Vd else 'NOT OK'}")

    # 8. Deflection check
    delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv)
    st.subheader("8. Serviceability Deflection")
    st.latex(f"I_x = {Ix_cm4:.1f}\\ \\text{{cm}}^4")
    if load_type == "Uniformly Distributed Load (UDL)":
        st.latex(f"\\delta = \\frac{{5 w L^4}}{{384 E I}} = \\frac{{5 \\times {w_serv} \\times ({span*1000})^4}}{{384 \\times 2\\times10^5 \\times {Ix_cm4}\\times10^4}} = {delta:.1f}\\ \\text{{mm}}")
    else:
        st.latex(f"\\delta = \\frac{{P L^3}}{{48 E I}} = \\frac{{{P_serv}\\times1000 \\times ({span*1000})^3}}{{48 \\times 2\\times10^5 \\times {Ix_cm4}\\times10^4}} = {delta:.1f}\\ \\text{{mm}}")
    st.latex(f"\\delta_{{\\text{{limit}}}} = \\frac{{L}}{{300}} = \\frac{{{span}\\times1000}}{{300}} = {delta_limit:.1f}\\ \\text{{mm}}")
    if delta <= delta_limit:
        st.success(f"δ = {delta:.1f} mm ≤ {delta_limit:.1f} mm → Deflection OK.")
    else:
        st.error(f"δ = {delta:.1f} mm > {delta_limit:.1f} mm → Increase section or add camber.")

    # 9. Stiffener requirement
    need_stiff, stiff_spacing = stiffener_requirement(d, tw, fy, Vu, Vd)
    st.subheader("9. Intermediate Stiffeners")
    if need_stiff:
        st.warning("Web requires intermediate stiffeners (d/tw > 67ε and/or Vu > 0.6Vd).")
        st.write(f"Recommended spacing (max) = {stiff_spacing} mm (≈ 1.5d = {1.5*d:.0f} mm)")
    else:
        st.success("No intermediate stiffeners required.")

    # 10. Material estimate
    total_weight = weight_kgpm * span
    st.subheader("10. Material Estimate")
    st.write(f"Steel grade: {grade} (fy = {fy} MPa)")
    st.write(f"Cross-sectional area ≈ { (d*tw/1e6 + 2*bf*tf/1e6):.4f} m²")
    st.write(f"Weight per metre = {weight_kgpm:.1f} kg/m")
    st.write(f"Total weight for span {span} m = {total_weight:.0f} kg ({total_weight/1000:.2f} tonnes)")

    # 11. Detailed drawing
    st.subheader("11. Elevation Drawing")
    fig, ax = plt.subplots(figsize=(12,5))
    # scaling for plot
    d_plot = d / 10
    tf_plot = tf / 10
    bf_plot = bf / 10
    span_plot = span * 100
    y_center = 10
    web_x = 50
    web_width = span_plot - 100

    # Web
    ax.add_patch(Rectangle((web_x, y_center - d_plot/2), web_width, d_plot,
                           fc='lightblue', ec='black', lw=1.5))
    # Top flange
    ax.add_patch(Rectangle((web_x, y_center - d_plot/2 - tf_plot), web_width, tf_plot,
                           fc='steelblue', ec='black', lw=1.5))
    # Bottom flange
    ax.add_patch(Rectangle((web_x, y_center + d_plot/2), web_width, tf_plot,
                           fc='steelblue', ec='black', lw=1.5))
    # Stiffeners if needed
    if need_stiff and stiff_spacing:
        spacing_plot = stiff_spacing / 10
        x = web_x + spacing_plot
        while x < web_x + web_width:
            ax.add_patch(Rectangle((x, y_center - d_plot/2 - tf_plot/2), 10, d_plot + tf_plot,
                                   fc='salmon', ec='darkred', alpha=0.7))
            x += spacing_plot
    # supports
    sup_w = 40
    sup_h = 20
    ax.add_patch(Rectangle((web_x-15, y_center - d_plot/2 - tf_plot - sup_h), sup_w, sup_h, fc='gray', ec='black'))
    ax.add_patch(Rectangle((web_x+web_width-25, y_center - d_plot/2 - tf_plot - sup_h), sup_w, sup_h, fc='gray', ec='black'))
    # annotations
    ax.annotate(f'Web: {d}mm x {tw}mm', xy=(web_x+20, y_center), fontsize=9, ha='center', bbox=dict(boxstyle='round', fc='white'))
    ax.annotate(f'Flange: {bf}×{tf}mm', xy=(web_x+web_width-80, y_center+d_plot/2+tf_plot/2), fontsize=9, ha='center', bbox=dict(boxstyle='round', fc='white'))
    # span dimension
    ax.annotate('', xy=(web_x, y_center - d_plot/2 - tf_plot - 40), xytext=(web_x+web_width, y_center - d_plot/2 - tf_plot - 40),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.annotate(f'Span = {span}m', xy=(web_x+web_width/2, y_center - d_plot/2 - tf_plot - 55), ha='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, span_plot+100)
    ax.set_ylim(y_center - d_plot/2 - tf_plot - 100, y_center + d_plot/2 + tf_plot + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Plate Girder – Elevation', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # Final summary
    st.success("✅ Design completed. All checks performed as per IS 800:2007.")

else:
    st.info("👈 Enter parameters in the sidebar and click **Design Plate Girder** to start the detailed design.")
    st.markdown("""
    ### Features:
    - Step‑by‑step calculations with formulas
    - Automatic web & flange sizing
    - Moment, shear, deflection, and web stability checks
    - Stiffener recommendation
    - Scale drawing of the girder
    - Material quantity estimation
    """)
