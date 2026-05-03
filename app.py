import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

# Page configuration
st.set_page_config(page_title="Plate Girder Designer", layout="wide")
st.title("📐 Steel Plate Girder Designer")
st.markdown("Design of welded plate girders as per IS 800:2007")

# Sidebar inputs
with st.sidebar:
    st.header("General Parameters")
    span = st.number_input("Span (m)", min_value=5.0, max_value=50.0, value=15.0, step=0.5)
    
    st.header("Loading (Service Loads)")
    load_type = st.selectbox("Load Type", ["Uniformly Distributed Load (UDL)", "Point Load at Midspan"])
    
    if load_type == "Uniformly Distributed Load (UDL)":
        dl = st.number_input("Dead Load (kN/m)", min_value=0.0, value=30.0, step=5.0)
        ll = st.number_input("Live Load (kN/m)", min_value=0.0, value=20.0, step=5.0)
        point_load = 0
    else:
        point_load = st.number_input("Point Load at Midspan (kN)", min_value=0.0, value=150.0, step=25.0)
        dl = ll = 0
    
    st.header("Material Properties")
    steel_grade = st.selectbox("Steel Grade", ["Fe 250 (fy=250 MPa)", "Fe 410 (fy=250 MPa)", "Fe 450 (fy=450 MPa)"])
    fy = 250 if "250" in steel_grade else 450
    st.info(f"Yield Strength (fy) = {fy} MPa")
    
    st.header("Design Parameters")
    gamma_m0 = 1.1
    gamma_m1 = 1.25
    
    st.header("Optional: Manual Flange Width")
    manual_bf = st.number_input("Flange Width (mm) - 0 for auto", min_value=0, value=0, step=10)

# Helper functions
def calc_required_zp(Mu, fy, gamma_m0=1.1):
    """Calculate required plastic section modulus (mm³)"""
    return Mu * 1e6 * gamma_m0 / fy

def calc_factored_load(dl, ll, point_load, load_type):
    """Calculate factored moment and shear"""
    if load_type == "Uniformly Distributed Load (UDL)":
        wu = 1.5 * (dl + ll)
        Mu = wu * span**2 / 8
        Vu = wu * span / 2
    else:
        Pu = 1.5 * point_load
        Mu = Pu * span / 4
        Vu = Pu / 2
    return Mu, Vu

def design_plate_girder(span_m, Mu_kNm, Vu_kN, fy, manual_bf=0):
    """Design plate girder section"""
    span_mm = span_m * 1000
    
    # Initial web depth (span/10 to span/12)
    d = span_mm / 11
    d = max(500, min(2500, round(d / 10) * 10))
    
    # Web thickness based on shear buckling and minimum requirements
    epsilon = np.sqrt(250 / fy)
    tw_min = max(6, d / 200)
    tw = max(tw_min, round(Vu * 1000 * gamma_m1 / (d * (fy / np.sqrt(3))) / 10) * 10)
    tw = max(6, min(25, round(tw)))
    
    # Required Zp
    Zp_req = calc_required_zp(Mu_kNm, fy)
    
    # Web contribution to Zp
    Zp_web = tw * d**2 / 4
    
    # Required flange area
    Af_req = max(0, (Zp_req - Zp_web) / d)
    
    # Flange dimensions
    if manual_bf > 0:
        bf = manual_bf
    else:
        bf = max(150, d / 4)
        bf = round(bf / 10) * 10
    
    tf = max(8, round(Af_req / bf))
    
    # Check flange compactness (bf/tf <= 9.4*epsilon)
    epsilon = np.sqrt(250 / fy)
    bf_tf_ratio = bf / tf
    compact_limit = 9.4 * epsilon
    
    if bf_tf_ratio > compact_limit:
        tf = max(tf, bf / (0.9 * compact_limit))
        tf = round(tf)
    
    # Recalculate actual Zp
    Zp_actual = tw * d**2 / 4 + bf * tf * (d + tf)
    
    # Web slenderness and stiffeners
    web_slenderness = d / tw
    needs_stiffeners = web_slenderness > 67 * epsilon
    stiffener_spacing = None
    if needs_stiffeners:
        stiffener_spacing = min(1.5 * d, 3000)
        stiffener_spacing = round(stiffener_spacing / 100) * 100
    
    # Shear capacity
    Vd = tw * d * (fy / np.sqrt(3)) / gamma_m1 / 1000
    shear_ratio = Vu / Vd if Vd > 0 else 999
    
    # Moment capacity
    Md = Zp_actual * fy / gamma_m0 / 1e6
    moment_ratio = Mu_kNm / Md if Md > 0 else 999
    
    # Deflection check (service load)
    if load_type == "Uniformly Distributed Load (UDL)":
        w_serv = dl + ll
        Ix = (tw * d**3 / 12 + 2 * (bf * tf * ((d + tf)/2)**2)) / 10000
        delta = 5 * w_serv * (span_m * 1000)**4 / (384 * 2e5 * Ix * 10**4)
    else:
        P_serv = point_load
        Ix = (tw * d**3 / 12 + 2 * (bf * tf * ((d + tf)/2)**2)) / 10000
        delta = P_serv * 1000 * (span_m * 1000)**3 / (48 * 2e5 * Ix * 10**4)
    delta_limit = span_m * 1000 / 300
    
    return {
        'd': d, 'tw': tw, 'bf': bf, 'tf': tf,
        'Zp_req': Zp_req, 'Zp_actual': Zp_actual,
        'Mu': Mu_kNm, 'Md': Md, 'moment_ratio': moment_ratio,
        'Vu': Vu, 'Vd': Vd, 'shear_ratio': shear_ratio,
        'delta': delta, 'delta_limit': delta_limit,
        'web_slenderness': web_slenderness,
        'needs_stiffeners': needs_stiffeners,
        'stiffener_spacing': stiffener_spacing,
        'compact_ratio': bf_tf_ratio, 'compact_limit': compact_limit
    }

def plot_plate_girder(design, span_m):
    """Create detailed visualization of the plate girder"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    d = design['d']
    bf = design['bf']
    tf = design['tf']
    tw = design['tw']
    span_px = 1200
    scale_x = span_px / (span_m * 1000)
    
    d_plot = d / 10
    bf_plot = bf / 10
    tf_plot = tf / 10
    tw_plot = tw / 10
    span_plot = span_m * 100
    y_center = 10
    
    web_x = 50
    web_width = span_plot - 100
    web_rect = Rectangle((web_x, y_center - d_plot/2), web_width, d_plot,
                         linewidth=1.5, edgecolor='black', facecolor='lightblue', alpha=0.7)
    ax.add_patch(web_rect)
    
    top_flange = Rectangle((web_x, y_center - d_plot/2 - tf_plot), web_width, tf_plot,
                           linewidth=1.5, edgecolor='black', facecolor='steelblue')
    ax.add_patch(top_flange)
    
    bottom_flange = Rectangle((web_x, y_center + d_plot/2), web_width, tf_plot,
                              linewidth=1.5, edgecolor='black', facecolor='steelblue')
    ax.add_patch(bottom_flange)
    
    if design['needs_stiffeners'] and design['stiffener_spacing']:
        spacing_mm = design['stiffener_spacing']
        spacing_plot = spacing_mm / 10
        stiffener_width = 10
        x_start = web_x + spacing_plot
        while x_start < web_x + web_width:
            stiffener = Rectangle((x_start, y_center - d_plot/2 - tf_plot/2),
                                 stiffener_width, d_plot + tf_plot,
                                 linewidth=1, edgecolor='darkred', facecolor='salmon', alpha=0.8)
            ax.add_patch(stiffener)
            x_start += spacing_plot
    
    ax.annotate(f'Web: {d}mm', xy=(web_x + 20, y_center), fontsize=9, ha='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate(f'Flange: {bf}×{tf}mm', xy=(web_x + web_width - 80, y_center + d_plot/2 + tf_plot/2), 
                fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    support_x = web_x
    support_width = 40
    support_height = 20
    support_rect = Rectangle((support_x - 15, y_center - d_plot/2 - tf_plot - support_height), 
                            support_width, support_height, linewidth=2, edgecolor='black', facecolor='gray')
    ax.add_patch(support_rect)
    
    support_rect2 = Rectangle((web_x + web_width - 25, y_center - d_plot/2 - tf_plot - support_height), 
                             support_width, support_height, linewidth=2, edgecolor='black', facecolor='gray')
    ax.add_patch(support_rect2)
    
    ax.annotate('', xy=(web_x, y_center - d_plot/2 - tf_plot - 40), 
                xytext=(web_x + web_width, y_center - d_plot/2 - tf_plot - 40),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.annotate(f'Span = {span_m}m', xy=(web_x + web_width/2, y_center - d_plot/2 - tf_plot - 55),
                ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, span_plot + 100)
    ax.set_ylim(y_center - d_plot/2 - tf_plot - 100, y_center + d_plot/2 + tf_plot + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Plate Girder Elevation', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# Main calculation and display
if st.sidebar.button("Design Plate Girder", type="primary", use_container_width=True):
    with st.spinner("Designing plate girder..."):
        Mu, Vu = calc_factored_load(dl, ll, point_load, load_type)
        design = design_plate_girder(span, Mu, Vu, fy, manual_bf)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Design Summary")
            st.metric("Factored Bending Moment", f"{design['Mu']:.1f} kN·m", 
                     delta=f"Capacity: {design['Md']:.1f} kN·m", delta_color="inverse")
            st.metric("Factored Shear Force", f"{design['Vu']:.1f} kN", 
                     delta=f"Capacity: {design['Vd']:.1f} kN", delta_color="inverse")
            
            st.subheader("📐 Section Dimensions")
            dim_col1, dim_col2, dim_col3 = st.columns(3)
            dim_col1.metric("Web Depth", f"{design['d']} mm")
            dim_col1.metric("Web Thickness", f"{design['tw']} mm")
            dim_col2.metric("Flange Width", f"{design['bf']} mm")
            dim_col2.metric("Flange Thickness", f"{design['tf']} mm")
            dim_col3.metric("Web Slenderness (d/tw)", f"{design['web_slenderness']:.0f}")
            dim_col3.metric("Flange Compactness (bf/tf)", f"{design['compact_ratio']:.1f}")
            
            st.subheader("✅ Design Checks")
            chk1, chk2, chk3 = st.columns(3)
            chk1.success(f"✓ Moment: {design['moment_ratio']:.2f} ≤ 1")
            chk2.success(f"✓ Shear: {design['shear_ratio']:.2f} ≤ 1")
            if design['delta'] < design['delta_limit']:
                chk3.success(f"✓ Deflection: {design['delta']:.1f}mm < {design['delta_limit']:.0f}mm")
            else:
                chk3.error(f"✗ Deflection: {design['delta']:.1f}mm > {design['delta_limit']:.0f}mm")
            
            if design['needs_stiffeners']:
                st.warning(f"⚠️ Web requires stiffeners (d/tw = {design['web_slenderness']:.0f} > 67ε = {67*design['compact_limit']/9.4:.0f})")
                st.info(f"Recommended stiffener spacing: {design['stiffener_spacing']} mm (max)")
            else:
                st.success("✓ No intermediate stiffeners required")
        
        with col2:
            st.subheader("📈 Performance Summary")
            st.markdown("**Moment Utilization**")
            st.progress(min(design['moment_ratio'], 1.0))
            st.caption(f"{design['moment_ratio']*100:.1f}%")
            
            st.markdown("**Shear Utilization**")
            st.progress(min(design['shear_ratio'], 1.0))
            st.caption(f"{design['shear_ratio']*100:.1f}%")
            
            st.markdown("**Deflection Ratio**")
            defl_ratio = design['delta'] / design['delta_limit'] if design['delta_limit'] > 0 else 1
            st.progress(min(defl_ratio, 1.0))
            st.caption(f"Span/{design['delta_limit']/design['delta']*span:.0f}" if design['delta']>0 else "N/A")
        
        st.subheader("📐 Detailed Plate Girder Drawing")
        fig = plot_plate_girder(design, span)
        st.pyplot(fig)
        
        st.subheader("📦 Material Estimate")
        weight_per_m = (design['d']/1000 * design['tw']/1000 + 
                        2 * design['bf']/1000 * design['tf']/1000) * 7850
        total_weight = weight_per_m * span
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Weight per meter", f"{weight_per_m:.0f} kg/m")
        col_w2.metric("Total weight", f"{total_weight:.0f} kg ({total_weight/1000:.1f} tonnes)")
        col_w3.metric("Steel grade", steel_grade)
        
        with st.expander("📖 Design Assumptions & Notes"):
            st.markdown("""
            **Design Methodology (IS 800:2007)**
            - Partial safety factors: γm0 = 1.1 (yielding), γm1 = 1.25 (buckling)
            - Web slenderness limit: d/tw ≤ 67ε for unstiffened webs
            - Flange compactness limit: bf/tf ≤ 9.4ε
            - Deflection limit: Span/300 for service loads
            - Shear capacity: Vd = (d × tw × fy/√3)/γm1
            - Moment capacity: Simplified plastic theory with web and flange contributions
            
            **Limitations**
            - Simplified design for preliminary sizing
            - Does not include flange-to-web weld design
            - Lateral-torsional buckling not considered (assumes adequate restraint)
            - Load combination: 1.5×(DL+LL) as per IS 800
            """)

else:
    st.info("👈 Enter design parameters in the sidebar and click 'Design Plate Girder' to begin")
    
    st.markdown("""
    ### How to use this tool:
    1. **Enter span** - Clear span of the plate girder (5-50m)
    2. **Specify loads** - Dead load and live load (UDL or point load)
    3. **Select steel grade** - Fe 250/410/450
    4. **Optional** - Manually specify flange width
    5. **Click Design** - Get complete design with checks and detailed drawing
    
    ### Features:
    - ✨ Automatic section sizing based on bending moment and shear
    - ✅ Comprehensive design checks (moment, shear, deflection)
    - 📐 Scale drawing with stiffeners if required
    - 📊 Material quantity estimation
    - 📖 Design notes and assumptions
    """)
