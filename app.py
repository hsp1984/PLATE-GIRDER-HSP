import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math

# Page configuration
st.set_page_config(page_title="Plate Girder Designer - IS 800:2007", layout="wide")
st.title("🏗️ Steel Plate Girder Designer")
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
    st.info(f"Yield strength (fy) = **{fy} MPa**")
    
    # Optional manual flange width
    st.header("3. Optional (Advanced)")
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

def required_plastic_modulus(Mu, fy, gamma_m0=1.10):
    """Required plastic section modulus Zp_req (mm³) as per Cl. 8.2.1.2"""
    return Mu * 1e6 * gamma_m0 / fy

def design_web(d, Vu, fy, gamma_m1=1.25):
    """Determine web thickness (mm) based on shear strength (Cl. 8.4.2.2)."""
    # Minimum thickness from serviceability (d/200) and absolute min 6 mm
    tw_min = max(6.0, d / 200.0)
    # Required thickness from shear capacity: Vd = (tw * d * (fy/√3)) / γm1  (kN)
    req_tw_shear = (Vu * 1000 * gamma_m1) / (d * (fy / math.sqrt(3)))
    tw = max(tw_min, req_tw_shear)
    # Round to nearest even mm for practical fabrication
    tw = max(6.0, min(40.0, round(tw / 2) * 2))
    return tw

def check_web_slenderness(d, tw, fy):
    """Check unstiffened web slenderness per Cl. 8.6.1.1."""
    epsilon = math.sqrt(250 / fy)
    limit = 67 * epsilon
    actual = d / tw
    return actual <= limit, limit, actual

def design_flanges(d, tw, Zp_req, fy, manual_bf):
    """Design flange width bf (mm) and thickness tf (mm) using plastic section modulus."""
    # Web contribution to Zp
    Zp_web = tw * d**2 / 4
    # Required flange area (assuming flanges carry the balance)
    Af_req = max(0.0, (Zp_req - Zp_web) / d)
    
    # Flange width
    if manual_bf > 0:
        bf = manual_bf
    else:
        bf = max(180.0, d / 4.0)
        bf = round(bf / 10) * 10
    
    tf = max(10.0, Af_req / bf)
    
    # Check flange compactness (Cl. 8.6.2): bf/tf ≤ 9.4 ε (for outstand elements)
    epsilon = math.sqrt(250 / fy)
    compact_limit = 9.4 * epsilon
    if (bf / tf) > compact_limit:
        tf = bf / (0.9 * compact_limit)   # tightening flange thickness
        tf = max(10.0, round(tf))
    
    tf = round(tf, 1)
    # For practical reasons, round tf to nearest 2 mm if < 40 mm, else 5 mm
    if tf < 40:
        tf = round(tf / 2) * 2
    else:
        tf = round(tf / 5) * 5
    
    return bf, tf, Zp_web

def compute_section_properties(d, tw, bf, tf):
    """Compute actual plastic modulus Zp (mm³), moment of inertia Ix (cm⁴),
       and weight per metre (kg/m)."""
    # Plastic modulus (approx., full plastic stress distribution)
    Zp_actual = tw * d**2 / 4 + bf * tf * (d + tf)
    # Elastic moment of inertia (cm⁴) for deflection
    Ix_cm4 = (tw * d**3 / 12 + 2 * (bf * tf * ((d + tf)/2)**2)) / 10000.0
    # Cross-sectional area (m²) and weight (kg/m) with steel density 7850 kg/m³
    area_m2 = (d/1000)*(tw/1000) + 2*(bf/1000)*(tf/1000)
    weight = area_m2 * 7850
    return Zp_actual, Ix_cm4, weight

def moment_capacity(Zp_actual, fy, gamma_m0=1.10):
    """Design moment capacity Md (kNm) as per Cl. 8.2.1.2"""
    return Zp_actual * fy / gamma_m0 / 1e6

def shear_capacity(d, tw, fy, gamma_m1=1.25):
    """Design shear capacity Vd (kN) as per Cl. 8.4.2.2 (simple post-critical method)"""
    return tw * d * (fy / math.sqrt(3)) / gamma_m1 / 1000.0

def deflection_check(span_m, Ix_cm4, w_serv, P_serv, load_type, E=2.0e5):
    """Compute deflection (mm) and limit (span/300 per Cl. 5.6.1)."""
    delta_limit = span_m * 1000 / 300.0
    if load_type == "Uniformly Distributed Load (UDL)":
        delta = 5.0 * w_serv * (span_m * 1000)**4 / (384 * E * Ix_cm4 * 1e4)
    else:
        delta = P_serv * 1000 * (span_m * 1000)**3 / (48 * E * Ix_cm4 * 1e4)
    return delta, delta_limit

def stiffener_requirements(d, tw, fy, Vu, Vd):
    """Determine need for intermediate stiffeners (Cl. 8.6.2 / 8.7.3)."""
    epsilon = math.sqrt(250 / fy)
    web_slenderness = d / tw
    # Clause 8.6.1.1: unstiffened web permitted only if d/tw ≤ 67ε
    if web_slenderness <= 67 * epsilon:
        return False, None
    # If capacity is insufficient or web is slender, stiffeners needed
    # Recommended spacing approx. 1.5d (max), but not more than 3000 mm (Cl. 8.7.3)
    spacing = min(1.5 * d, 3000)
    spacing = round(spacing / 100) * 100
    return True, spacing

def bearing_stiffeners_required(Vu, d, tw, fy, gamma_m0=1.10):
    """Quick check: end bearing stiffener generally required for concentrated reaction."""
    # As per Cl. 8.7.4, bearing stiffeners are required if the web alone cannot resist
    # the reaction. For simplicity, we always recommend them in design.
    # Here we return boolean and area requirement.
    bearing_capacity = (tw * (d/2) * (fy / math.sqrt(3))) / gamma_m0 / 1000  # kN (approx)
    return bearing_capacity < Vu, bearing_capacity

# ---------- Main Design Execution ----------
if st.sidebar.button("🚀 Design Plate Girder", type="primary", use_container_width=True):
    
    # 1. Factored Loads
    Mu, Vu, w_serv, P_serv = calc_factored_loads(dl, ll, point_load, span, load_type)
    
    st.header("📊 Design Calculations (Step by Step)")
    
    # Step 2: Required plastic modulus
    Zp_req = required_plastic_modulus(Mu, fy)
    
    # Step 3: Initial web depth
    d_initial = span * 1000 / 11   # economical depth range span/10 to span/12
    d = max(500, min(2500, round(d_initial / 10) * 10))
    
    # Step 4: Web thickness
    tw = design_web(d, Vu, fy)
    
    # Step 5: Flange sizing
    bf, tf, Zp_web = design_flanges(d, tw, Zp_req, fy, manual_bf)
    
    # Step 6: Actual section properties
    Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
    Md = moment_capacity(Zp_actual, fy)
    Vd = shear_capacity(d, tw, fy)
    
    # Step 7: Web slenderness check
    web_ok, web_limit, web_actual = check_web_slenderness(d, tw, fy)
    
    # Step 8: Stiffener checks
    need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Vd)
    need_bear_stiff, bearing_cap = bearing_stiffeners_required(Vu, d, tw, fy)
    
    # Step 9: Deflection
    delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
    
    # Step 10: Display all calculations
    # -------------------------------------------------
    st.subheader("1️⃣ Loads & Factored Moments")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Span:** `{span} m`")
        if load_type == "Uniformly Distributed Load (UDL)":
            st.latex(f"w_{{service}} = DL + LL = {dl} + {ll} = {dl+ll:.2f}\\ \\text{{kN/m}}")
            st.latex(f"w_u = 1.5 \\times {dl+ll:.2f} = {1.5*(dl+ll):.2f}\\ \\text{{kN/m}}")
            st.latex(f"M_u = \\frac{{w_u L^2}}{{8}} = \\frac{{{1.5*(dl+ll):.2f} \\times {span}^2}}{{8}} = {Mu:.1f}\\ \\text{{kN·m}}")
            st.latex(f"V_u = \\frac{{w_u L}}{{2}} = \\frac{{{1.5*(dl+ll):.2f} \\times {span}}}{{2}} = {Vu:.1f}\\ \\text{{kN}}")
        else:
            st.latex(f"P_{{service}} = {point_load}\\ \\text{{kN}}")
            st.latex(f"P_u = 1.5 \\times {point_load} = {1.5*point_load:.2f}\\ \\text{{kN}}")
            st.latex(f"M_u = \\frac{{P_u L}}{{4}} = \\frac{{{1.5*point_load:.2f} \\times {span}}}{{4}} = {Mu:.1f}\\ \\text{{kN·m}}")
            st.latex(f"V_u = \\frac{{P_u}}{{2}} = \\frac{{{1.5*point_load:.2f}}}{{2}} = {Vu:.1f}\\ \\text{{kN}}")
    
    with col2:
        st.metric("Factored Moment (Mu)", f"{Mu:.1f} kN·m")
        st.metric("Factored Shear (Vu)", f"{Vu:.1f} kN")
    
    st.subheader("2️⃣ Required Plastic Modulus")
    st.latex(f"Z_{{p,\\text{{req}}}} = \\frac{{M_u \\cdot \\gamma_{{m0}}}}{{f_y}} = \\frac{{{Mu:.1f}\\times10^6 \\times 1.1}}{{{fy}}} = {Zp_req:.0f}\\ \\text{{mm}}^3")
    
    st.subheader("3️⃣ Web Design (Cl. 8.6.1 & 8.4.2.2)")
    st.latex(f"\\text{{Trial depth: }} d \\approx \\frac{{L}}{{11}} = \\frac{{{span}\\times1000}}{{11}} = {d_initial:.0f}\\ \\text{{mm}} \\rightarrow \\text{{Adopted }} d = {d}\\ \\text{{mm}}")
    st.latex(f"t_w \\ge \\frac{{V_u \\gamma_{{m1}}}}{{d \\cdot (f_y/\\sqrt{{3}})}} = \\frac{{{Vu}\\times1000 \\times 1.25}}{{{d} \\times ({fy}/\\sqrt{{3}})}} = {design_web(d, Vu, fy):.1f}\\ \\text{{mm}}")
    st.latex(f"t_w \\ge \\frac{{d}}{{200}} = \\frac{{{d}}}{{200}} = {d/200:.1f}\\ \\text{{mm}}")
    st.success(f"✅ **Selected web thickness:** `{tw} mm`")
    
    st.subheader("4️⃣ Web Slenderness (Unstiffened)")
    st.latex(f"\\frac{{d}}{{t_w}} = \\frac{{{d}}}{{{tw}}} = {web_actual:.1f}")
    epsilon_val = math.sqrt(250 / fy)
    st.latex(f"\\varepsilon = \\sqrt{{\\frac{{250}}{{f_y}}}} = \\sqrt{{\\frac{{250}}{{{fy}}}}} = {epsilon_val:.3f}")
    st.latex(f"\\text{{Limit for unstiffened web: }} \\frac{{d}}{{t_w}} \\le 67\\varepsilon = 67 \\times {epsilon_val:.3f} = {web_limit:.1f}")
    if web_ok:
        st.success(f"✓ d/tw = {web_actual:.1f} ≤ {web_limit:.1f} → unstiffened web is acceptable.")
        need_stiff = False
    else:
        st.warning(f"⚠ d/tw = {web_actual:.1f} > {web_limit:.1f} → intermediate stiffeners are required.")
    
    st.subheader("5️⃣ Flange Design (Plastic Section Modulus)")
    st.latex(f"Z_{{p,\\text{{web}}}} = \\frac{{t_w d^2}}{{4}} = \\frac{{{tw} \\times {d}^2}}{{4}} = {Zp_web:.0f}\\ \\text{{mm}}^3")
    Af_req = max(0, (Zp_req - Zp_web) / d)
    st.latex(f"A_{{f,\\text{{req}}}} = \\frac{{Z_{{p,\\text{{req}}}} - Z_{{p,\\text{{web}}}}}}{{d}} = \\frac{{{Zp_req:.0f} - {Zp_web:.0f}}}{{{d}}} = {Af_req:.0f}\\ \\text{{mm}}^2")
    st.latex(f"b_f \\text{{ (from economy) }} \\approx \\frac{{d}}{{4}} = {d/4:.0f}\\ \\text{{mm}} \\rightarrow \\text{{Adopted }} b_f = {bf:.0f}\\ \\text{{mm}}")
    st.latex(f"t_f = \\frac{{A_{{f,\\text{{req}}}}}}{{b_f}} = \\frac{{{Af_req:.0f}}}{{{bf:.0f}}} = {tf:.1f}\\ \\text{{mm}}")
    st.success(f"✅ **Selected flanges:** `{bf:.0f} mm × {tf:.1f} mm`")
    # Compactness check
    compact_limit = 9.4 * epsilon_val
    bf_tf = bf / tf
    st.latex(f"\\text{{Flange compactness: }} \\frac{{b_f}}{{t_f}} = {bf_tf:.1f} \\le 9.4\\varepsilon = {compact_limit:.1f}")
    if bf_tf <= compact_limit:
        st.success("✓ Flange is compact (Class 1/2).")
    else:
        st.error("⚠ Flange is slender — increase tf or reduce bf.")
    
    st.subheader("6️⃣ Section Capacity (Cl. 8.2.1.2 & 8.4.2.2)")
    st.latex(f"Z_{{p,\\text{{actual}}}} = {Zp_actual:.0f}\\ \\text{{mm}}^3")
    st.latex(f"M_d = \\frac{{Z_p f_y}}{{\\gamma_{{m0}}}} = \\frac{{{Zp_actual:.0f} \\times {fy}}}{{1.1\\times10^6}} = {Md:.2f}\\ \\text{{kN·m}}")
    st.latex(f"V_d = \\frac{{t_w d (f_y/\\sqrt{{3}})}}{{\\gamma_{{m1}}}} = \\frac{{{tw}\\times{d}\\times({fy}/\\sqrt{{3}})}}{{1.25\\times1000}} = {Vd:.2f}\\ \\text{{kN}}")
    
    col1, col2 = st.columns(2)
    with col1:
        ratio_moment = Mu / Md if Md > 0 else 999
        st.metric("Moment ratio", f"{ratio_moment:.3f}", help="Should be ≤ 1.0")
        if ratio_moment <= 1.0:
            st.success("✓ Moment capacity OK")
        else:
            st.error("✗ Insufficient moment capacity — increase section")
    with col2:
        ratio_shear = Vu / Vd if Vd > 0 else 999
        st.metric("Shear ratio", f"{ratio_shear:.3f}", help="Should be ≤ 1.0")
        if ratio_shear <= 1.0:
            st.success("✓ Shear capacity OK")
        else:
            st.error("✗ Insufficient shear capacity — increase tw or add stiffeners")
    
    st.subheader("7️⃣ Serviceability: Deflection (Cl. 5.6.1)")
    st.latex(f"I_x = {Ix_cm4:.1f}\\ \\text{{cm}}^4")
    if load_type == "Uniformly Distributed Load (UDL)":
        st.latex(f"\\delta = \\frac{{5 w L^4}}{{384 E I}} = \\frac{{5 \\times {w_serv:.2f} \\times ({span*1000})^4}}{{384 \\times 2\\times10^5 \\times {Ix_cm4:.1f}\\times10^4}} = {delta:.1f}\\ \\text{{mm}}")
    else:
        st.latex(f"\\delta = \\frac{{P L^3}}{{48 E I}} = \\frac{{{P_serv:.2f}\\times1000 \\times ({span*1000})^3}}{{48 \\times 2\\times10^5 \\times {Ix_cm4:.1f}\\times10^4}} = {delta:.1f}\\ \\text{{mm}}")
    st.latex(f"\\delta_{{\\text{{limit}}}} = \\frac{{L}}{{300}} = \\frac{{{span}\\times1000}}{{300}} = {delta_limit:.1f}\\ \\text{{mm}}")
    if delta <= delta_limit:
        st.success(f"✓ Deflection ok: {delta:.1f} mm ≤ {delta_limit:.1f} mm")
    else:
        st.warning(f"⚠ High deflection: {delta:.1f} mm > {delta_limit:.1f} mm — consider larger stiffness.")
    
    st.subheader("8️⃣ Web Stiffener Requirements")
    if need_stiff:
        st.warning(f"⚠ Intermediate stiffeners required at max spacing ≈ {stiff_spacing} mm (≤ 1.5d).")
        st.info(f"📐 Recommended stiffener spacing: **{stiff_spacing} mm** (as per Cl. 8.7.3)")
    else:
        st.success("✅ No intermediate stiffeners required.")
    
    if need_bear_stiff:
        st.warning("⚠ End bearing stiffeners are required — design per Cl. 8.7.4.")
    else:
        st.info("End bearing stiffeners may still be provided as a conservative measure.")
    
    st.subheader("9️⃣ Material Estimate")
    total_weight = weight * span
    st.metric("Weight per meter", f"{weight:.0f} kg/m")
    st.metric("Total weight", f"{total_weight:.0f} kg ({total_weight/1000:.2f} tonnes)")
    
    # ---------- Draw Elevation with Stiffeners ----------
    st.subheader("🔟 Detailed Elevation Drawing")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Scale for visualisation
    d_plot = d / 10      # cm
    tf_plot = tf / 10
    span_plot = span * 100  # cm
    y_center = 10
    web_x = 40
    web_width = span_plot - 80
    
    # Web rectangle
    web_rect = Rectangle((web_x, y_center - d_plot/2), web_width, d_plot,
                         linewidth=1.5, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(web_rect)
    
    # Top flange
    top_flange = Rectangle((web_x, y_center - d_plot/2 - tf_plot), web_width, tf_plot,
                           linewidth=1.5, edgecolor='black', facecolor='steelblue')
    ax.add_patch(top_flange)
    
    # Bottom flange
    bottom_flange = Rectangle((web_x, y_center + d_plot/2), web_width, tf_plot,
                              linewidth=1.5, edgecolor='black', facecolor='steelblue')
    ax.add_patch(bottom_flange)
    
    # Intermediate stiffeners if required
    if need_stiff and stiff_spacing:
        spacing_plot = stiff_spacing / 10
        stiff_width = 8   # mm in plot units (0.8 cm)
        x_start = web_x + spacing_plot
        while x_start < web_x + web_width:
            stiff_rect = Rectangle((x_start, y_center - d_plot/2 - tf_plot/2),
                                   stiff_width, d_plot + tf_plot,
                                   linewidth=1, edgecolor='darkred', facecolor='salmon', alpha=0.8)
            ax.add_patch(stiff_rect)
            x_start += spacing_plot
    
    # End bearing stiffeners (schematically)
    end_stiff_width = 12
    end_stiff = Rectangle((web_x - end_stiff_width/2, y_center - d_plot/2 - tf_plot/2),
                          end_stiff_width, d_plot + tf_plot,
                          linewidth=1.5, edgecolor='brown', facecolor='peru', alpha=0.9)
    ax.add_patch(end_stiff)
    end_stiff_right = Rectangle((web_x + web_width - end_stiff_width/2, y_center - d_plot/2 - tf_plot/2),
                                end_stiff_width, d_plot + tf_plot,
                                linewidth=1.5, edgecolor='brown', facecolor='peru', alpha=0.9)
    ax.add_patch(end_stiff_right)
    
    # Annotations
    ax.annotate(f'Web: {d}mm × {tw}mm', xy=(web_x + 25, y_center), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate(f'Flange: {bf}×{tf}mm', xy=(web_x + web_width - 50, y_center + d_plot/2 + tf_plot/2),
                fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if need_stiff:
        ax.annotate('Intermediate\nstiff', xy=(web_x + web_width/2, y_center - d_plot/2 - tf_plot),
                    fontsize=8, ha='center', color='darkred')
    ax.annotate('Bearing\nstiff', xy=(web_x - 15, y_center - d_plot/4), fontsize=8, ha='center', color='brown')
    
    # Span dimension line
    ax.annotate('', xy=(web_x, y_center - d_plot/2 - tf_plot - 40),
                xytext=(web_x + web_width, y_center - d_plot/2 - tf_plot - 40),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.annotate(f'Span = {span} m', xy=(web_x + web_width/2, y_center - d_plot/2 - tf_plot - 55),
                ha='center', fontsize=11, fontweight='bold')
    
    # Supports
    support_width = 30
    support_height = 20
    support_left = Rectangle((web_x - 18, y_center - d_plot/2 - tf_plot - support_height),
                             support_width, support_height,
                             linewidth=2, edgecolor='black', facecolor='gray')
    ax.add_patch(support_left)
    support_right = Rectangle((web_x + web_width - 12, y_center - d_plot/2 - tf_plot - support_height),
                              support_width, support_height,
                              linewidth=2, edgecolor='black', facecolor='gray')
    ax.add_patch(support_right)
    
    ax.set_xlim(0, span_plot + 120)
    ax.set_ylim(y_center - d_plot/2 - tf_plot - 100, y_center + d_plot/2 + tf_plot + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Plate Girder Elevation', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # Final summary
    st.success("✅ Design completed. All checks performed as per IS 800:2007.")
    st.markdown("---")
    st.caption("Note: The design follows limit state method. Intermediate and bearing stiffeners are shown schematically. For detailed welding and connection design, refer to IS 800:2007, Cl. 8.7 and 10.5.")

else:
    st.info("👈 Enter design parameters in the sidebar and click **Design Plate Girder** to start.")
    st.markdown("""
    ### 📖 Design Steps (As per IS 800:2007)
    1. **Load calculations** – Factored bending moment and shear force
    2. **Required plastic modulus** \( Z_p \) (Cl. 8.2.1.2)
    3. **Web depth and thickness** – Shear capacity and buckling checks (Cl. 8.6.1, 8.4.2)
    4. **Flange sizing** – Plastic modulus and compactness (Cl. 8.6.2)
    5. **Moment / shear capacity** – Limit state checks
    6. **Deflection** – Serviceability limit (Cl. 5.6.1)
    7. **Stiffener design** – Intermediate (Cl. 8.7.3) and bearing (Cl. 8.7.4)
    8. **Elevation drawing** – Automatic plot with all dimensions
    """)
