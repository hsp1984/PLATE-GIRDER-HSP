import streamlit as st
import numpy as np
import math

st.set_page_config(page_title="Plate Girder Designer - IS 800:2007", layout="wide")
st.title("🏗️ Steel Plate Girder Designer (Shear Buckling + Auto‑Revision)")
st.markdown("**👨‍🏫 Developer:** *Dr Hiteshkumar Santosh Patil, Assistant Professor, Email: hitesh.patil@rcpit.ac.in, Civil Engineering Department, RCPIT SHIRPUR**")
st.markdown("Design as per **IS 800:2007 (Limit State Method)**")
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

def shear_buckling_strength(d, tw, fy, has_stiffeners=False, stiff_spacing=None, tension_field=False):
    epsilon = math.sqrt(250 / fy)
    if has_stiffeners and stiff_spacing:
        c = stiff_spacing
        cd_ratio = c / d
        if cd_ratio < 1.0:
            k_v = 5.35 + 4.0 / (cd_ratio)**2
        else:
            k_v = 4.0 + 5.35 / (cd_ratio)**2
    else:
        k_v = 5.35
    λ_w = (d / tw) / (37.4 * epsilon * math.sqrt(k_v))
    Av = d * tw
    if λ_w <= 0.8:
        τ_b = fy / math.sqrt(3)
        method = "Simple post‑critical (λ_w ≤ 0.8)"
    elif 0.8 < λ_w < 1.2:
        τ_b = (1 - 0.8*(λ_w - 0.8)) * fy / math.sqrt(3)
        method = "Simple post‑critical (0.8 < λ_w < 1.2)"
    else:
        τ_b = fy / (math.sqrt(3) * λ_w**2)
        method = "Simple post‑critical (λ_w ≥ 1.2)"
    Vn = Av * τ_b / 1000
    if tension_field and has_stiffeners and λ_w > 1.0:
        Vtf = Av * (fy / math.sqrt(3)) * (1 - 1/λ_w**2) / 1000
        Vn += Vtf
        method += " + Tension field action (Cl. 8.4.2.2.3)"
    return Vn, method, λ_w

def check_web_slenderness_full(d, tw, fy, has_stiffeners=False, stiff_spacing=None):
    epsilon = math.sqrt(250 / fy)
    actual = d / tw
    if not has_stiffeners:
        limit = 67 * epsilon
        desc = f"Without transverse stiffeners: d/tw ≤ 67ε = {limit:.1f} (Cl. 8.6.1.1)"
    else:
        c = stiff_spacing if stiff_spacing else (1.5 * d)
        if c >= d:
            limit = 200 * epsilon
            desc = f"With transverse stiffeners (c ≥ d): d/tw ≤ 200ε = {limit:.1f} (Cl. 8.6.1.1)"
        elif c >= 0.74 * d:
            limit = 200 * epsilon
            desc = f"With transverse stiffeners (0.74d ≤ c < d): d/tw ≤ 200ε = {limit:.1f} (Cl. 8.6.1.1)"
        else:
            limit = 270 * epsilon
            desc = f"With transverse stiffeners (c < d): d/tw ≤ 270ε = {limit:.1f} (Cl. 8.6.1.1)"
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

def shear_capacity(d, tw, fy, has_stiffeners=False, stiff_spacing=None, tension_field=False):
    Vn, method, λ_w = shear_buckling_strength(d, tw, fy, has_stiffeners, stiff_spacing, tension_field)
    Vd = Vn / 1.25
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

def design_intermediate_stiffeners(d, tw, fy):
    epsilon = math.sqrt(250 / fy)
    bs = max(round(d / 30), 50)
    bs = round(bs / 10) * 10
    ts = max(6.0, math.ceil(bs / (14 * epsilon)))
    ts = max(6, round(ts / 2) * 2)
    outstand_ok = bs <= 14 * epsilon * ts
    thickness_ok = ts >= 6
    return ts, bs, outstand_ok, thickness_ok

def design_bearing_stiffeners(d, tw, bf, tf, fy, Vu):
    epsilon = math.sqrt(250 / fy)
    bb = max(round(d / 30), round(bf / 2))
    bb = min(bb, 200)
    bb = round(bb / 10) * 10
    tb = max(10.0, math.ceil(bb / 20))
    tb = round(tb / 2) * 2
    area_stiff = 2 * bb * tb
    bearing_capacity = area_stiff * fy / 1.1 / 1000
    area_ok = bearing_capacity >= Vu
    outstand_ok = bb <= 14 * epsilon * tb
    weld_size = max(6, round(tb / 2))
    return tb, bb, area_ok, outstand_ok, weld_size, bearing_capacity

def design_longitudinal_stiffeners(d, fy):
    epsilon = math.sqrt(250 / fy)
    loc = 0.2 * d
    bl = max(round(d / 30), 50)
    bl = round(bl / 10) * 10
    tl = max(6.0, math.ceil(bl / (14 * epsilon)))
    tl = max(6, round(tl / 2) * 2)
    outstand_ok = bl <= 14 * epsilon * tl
    thickness_ok = tl >= 6
    return tl, bl, loc, outstand_ok, thickness_ok

# ---------- Main Design ----------
if st.sidebar.button("🚀 Design Plate Girder", type="primary", use_container_width=True):
    
    Mu, Vu, w_serv, P_serv = calc_factored_loads(dl, ll, point_load, span, load_type)
    Zp_req = required_plastic_modulus(Mu, fy)
    
    d_guess = economical_depth_iterative(Mu, fy, target_K=100)
    tw_temp = design_web(d_guess, Vu, fy)
    K_actual = d_guess / tw_temp
    d = economical_depth_iterative(Mu, fy, target_K=K_actual)
    d = max(400, min(3000, round(d / 10) * 10))
    tw = design_web(d, Vu, fy)
    
    bf, tf, _ = design_flanges(d, tw, Zp_req, fy, manual_bf)
    
    max_iter = 25
    iter_count = 0
    revised = False
    while iter_count < max_iter:
        Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
        Md = moment_capacity(Zp_actual, fy)
        need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Md)
        Vd, shear_method, λ_w = shear_capacity(d, tw, fy, need_stiff, stiff_spacing, use_tension_field)
        web_ok, web_limit, web_actual, web_desc = check_web_slenderness_full(d, tw, fy, need_stiff, stiff_spacing)
        shear_ok = Vu <= Vd
        moment_ok = Mu <= Md
        delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
        defl_ok = delta <= delta_limit
        
        ts, bs, outstand_ok_inter, thick_ok_inter = design_intermediate_stiffeners(d, tw, fy)
        tb, bb, area_ok_bear, outstand_ok_bear, weld_bear, bearing_cap = design_bearing_stiffeners(d, tw, bf, tf, fy, Vu)
        stiff_ok = (not need_stiff or (outstand_ok_inter and thick_ok_inter)) and area_ok_bear and outstand_ok_bear
        
        if web_ok and shear_ok and moment_ok and defl_ok and stiff_ok:
            break
        revised = True
        if not web_ok or not shear_ok:
            tw += 2
            tw = min(tw, 40)
        if not moment_ok or delta > 1.2 * delta_limit:
            tf += 2
            tf = min(tf, 60) if tf < 40 else tf + 5
        if web_ok and moment_ok and not defl_ok:
            d += 20
            d = min(d, 2500)
            tw = max(tw, design_web(d, Vu, fy))
        if need_stiff and not (outstand_ok_inter and thick_ok_inter):
            ts += 2
        if not (area_ok_bear and outstand_ok_bear):
            tb += 2
        bf, tf, _ = design_flanges(d, tw, Zp_req, fy, manual_bf)
        iter_count += 1
    
    Zp_actual, Ix_cm4, weight = compute_section_properties(d, tw, bf, tf)
    Md = moment_capacity(Zp_actual, fy)
    need_stiff, stiff_spacing = stiffener_requirements(d, tw, fy, Vu, Md)
    Vd, shear_method, λ_w = shear_capacity(d, tw, fy, need_stiff, stiff_spacing, use_tension_field)
    web_ok, web_limit, web_actual, web_desc = check_web_slenderness_full(d, tw, fy, need_stiff, stiff_spacing)
    delta, delta_limit = deflection_check(span, Ix_cm4, w_serv, P_serv, load_type)
    ratio_moment = Mu / Md
    ratio_shear = Vu / Vd
    
    ts, bs, outstand_ok_inter, thick_ok_inter = design_intermediate_stiffeners(d, tw, fy)
    tb, bb, area_ok_bear, outstand_ok_bear, weld_bear, bearing_cap = design_bearing_stiffeners(d, tw, bf, tf, fy, Vu)
    tl, bl, long_loc, outstand_ok_long, thick_ok_long = design_longitudinal_stiffeners(d, fy)
    
    st.header("📊 Design Calculations (Step by Step) – IS 800:2007")
    if revised:
        st.info(f"Section revised after {iter_count} iteration(s).")
    else:
        st.success("Initial design passed all checks.")
    
    with st.expander("📐 1. Factored Loads & Required Plastic Modulus (Cl. 5.3.3 & 8.2.1.2)", expanded=True):
        st.latex(r"\text{For UDL: } M_u = \frac{1.5 (DL+LL) L^2}{8},\quad V_u = \frac{1.5 (DL+LL) L}{2}")
        st.latex(r"\text{For point load: } M_u = \frac{1.5 P L}{4},\quad V_u = \frac{1.5 P}{2}")
        st.latex(fr"M_u = {Mu:.1f} \,kN\,m,\quad V_u = {Vu:.1f} \,kN")
        st.latex(fr"Z_{{p,req}} = \frac{{M_u \gamma_{{m0}}}}{{f_y}} = \frac{{{Mu:.1f}\times10^6 \times 1.1}}{{{fy}}} = {Zp_req:.0f} \,mm^3")
    
    with st.expander("📐 2. Economical Depth (Cl. 8.6)", expanded=True):
        st.latex(r"d = \left( \frac{M_u K}{f_y} \right)^{1/3},\quad K = d/t_w")
        st.latex(fr"K = {d/tw:.1f} \quad\Rightarrow\quad d = {d} \,mm")
    
    with st.expander("📐 3. Web Thickness (Cl. 8.4.2.2 & 8.6.1.1)", expanded=True):
        st.latex(r"t_w \ge \frac{V_u \gamma_{m1}}{d (f_y/\sqrt{3})},\quad t_w \ge d/200")
        st.latex(fr"t_w = {tw} \,mm")
    
    with st.expander("📐 4. Shear Buckling Resistance (Cl. 8.4.2.2)", expanded=True):
        st.latex(r"\lambda_w = \frac{d/t_w}{37.4\,\varepsilon\sqrt{k_v}},\quad \varepsilon = \sqrt{250/f_y}")
        st.latex(fr"\lambda_w = {λ_w:.3f}")
        st.latex(r"""
        \tau_b = \begin{cases}
        f_y/\sqrt{3} & \lambda_w \le 0.8 \\
        [1-0.8(\lambda_w-0.8)]\,f_y/\sqrt{3} & 0.8<\lambda_w<1.2 \\
        f_y/(\sqrt{3}\,\lambda_w^2) & \lambda_w\ge1.2
        \end{cases}
        """)
        st.latex(fr"\tau_b \text{{ from {shear_method}}}")
        Vn_temp = shear_buckling_strength(d, tw, fy, need_stiff, stiff_spacing, use_tension_field)[0]
        st.latex(fr"V_n = A_v \tau_b = {Vn_temp:.1f} \,kN")
        st.latex(fr"V_d = V_n / \gamma_{{m1}} = {Vd:.2f} \,kN")
    
    with st.expander("📐 5. Web Slenderness (Cl. 8.6.1.1)", expanded=True):
        st.latex(fr"d/t_w = {web_actual:.1f}")
        st.write(f"**Criteria:** {web_desc}")
        st.write(f"→ {'✓ OK' if web_ok else '✗ NOT OK'}")
    
    bf_tf_ratio = bf / tf
    epsilon_val = math.sqrt(250 / fy)
    compact_limit = 9.4 * epsilon_val
    with st.expander("📐 6. Flange Design & Section Capacity (Cl. 8.6.2 & 8.2.1.2)", expanded=True):
        Zp_web_val = tw * d**2 / 4
        st.latex(fr"Z_{{p,web}} = \frac{{t_w d^2}}{4} = {Zp_web_val:.0f} \,mm^3")
        Af_req_val = max(0, (Zp_req - Zp_web_val) / d)
        st.latex(fr"A_{{f,req}} = \frac{{Z_{{p,req}} - Z_{{p,web}}}}{d} \approx {Af_req_val:.0f} \,mm^2")
        st.latex(fr"b_f = {bf:.0f} \,mm,\quad t_f = {tf:.1f} \,mm")
        st.latex(fr"Z_{{p,actual}} = {Zp_actual:.0f} \,mm^3")
        st.latex(fr"M_d = \frac{{Z_p f_y}}{{\gamma_{{m0}}}} = {Md:.2f} \,kN\,m")
        st.latex(fr"b_f/t_f = {bf_tf_ratio:.1f} \le 9.4\varepsilon = {compact_limit:.1f} \Rightarrow {'compact' if bf_tf_ratio <= compact_limit else 'slender'}")
    
    with st.expander("📐 7. Deflection Calculation (Cl. 5.6.1)", expanded=True):
        if load_type == "Uniformly Distributed Load (UDL)":
            st.latex(r"\delta = \frac{5 w L^4}{384 E I_x},\quad \delta_{limit} = L/300")
        else:
            st.latex(r"\delta = \frac{P L^3}{48 E I_x},\quad \delta_{limit} = L/300")
        st.latex(fr"I_x = {Ix_cm4:.1f} \,cm^4")
        st.latex(fr"\delta = {delta:.1f} \,mm,\quad \delta_{{limit}} = {delta_limit:.1f} \,mm")
        st.write(f"→ {'✓ OK' if delta <= delta_limit else '✗ NOT OK'}")
    
    with st.expander("📐 8. Stiffener Design (Cl. 8.7.2, 8.7.3, 8.7.4)", expanded=True):
        st.subheader("8.1 Intermediate Transverse Stiffeners (Cl. 8.7.3)")
        if need_stiff:
            st.warning(f"Required at spacing ≤ {stiff_spacing} mm (≤ 1.5d = {1.5*d:.0f} mm).")
            st.write(f"**Proposed size:** {bs} mm × {ts} mm flat (double-sided)")
            st.latex(r"Outstand: b_s \le 14\, \varepsilon \, t_s")
            st.latex(fr"{bs} \le 14 \times {epsilon_val:.3f} \times {ts} = {14*epsilon_val*ts:.1f} \,mm \rightarrow {'✓ OK' if outstand_ok_inter else '✗ NOT OK'}")
        else:
            st.success("Not required (d/t_w ≤ 67ε).")
        
        st.subheader("8.2 End Bearing Stiffeners (Cl. 8.7.4)")
        st.write(f"**Proposed size:** Two flats of {bb} mm × {tb} mm (one on each side)")
        st.latex(r"Area check: 2 b_b t_b f_y / \gamma_{m0} \ge V_u")
        st.latex(fr"2 \times {bb} \times {tb} \times {fy} / 1.1 = {bearing_cap:.1f} \,kN \ge {Vu:.1f} \,kN \rightarrow {'✓ OK' if area_ok_bear else '✗ NOT OK'}")
        st.latex(r"Outstand: b_b \le 14\, \varepsilon \, t_b")
        st.latex(fr"{bb} \le 14 \times {epsilon_val:.3f} \times {tb} = {14*epsilon_val*tb:.1f} \,mm \rightarrow {'✓ OK' if outstand_ok_bear else '✗ NOT OK'}")
        st.write(f"Recommended weld leg = {weld_bear} mm (continuous double fillet)")
        
        st.subheader("8.3 Longitudinal Stiffeners (Cl. 8.7.2) – Optional")
        if d / tw > 200 * epsilon_val:
            st.info(f"d/t_w = {web_actual:.1f} > 200ε = {200*epsilon_val:.1f}. Longitudinal stiffener recommended.")
            st.write(f"**Location:** {long_loc:.0f} mm from compression flange")
            st.write(f"**Proposed size:** {bl} mm × {tl} mm flat")
            st.latex(fr"Outstand: {bl} \le 14 \times {epsilon_val:.3f} \times {tl} = {14*epsilon_val*tl:.1f} \,mm \rightarrow {'✓ OK' if outstand_ok_long else '✗ NOT OK'}")
        else:
            st.success("Not mandatory (d/t_w ≤ 200ε).")
    
    total_weight = weight * span
    st.subheader("💰 Material Estimate")
    col1, col2 = st.columns(2)
    col1.metric("Weight per meter", f"{weight:.0f} kg/m")
    col2.metric("Total weight", f"{total_weight:.0f} kg ({total_weight/1000:.2f} tonnes)")
    
    st.subheader("📋 Final Design Summary Table (After Auto‑Revision)")
    summary_data = {
        "Parameter": [
            "Span", "Load type", "Factored Moment (Mu)", "Factored Shear (Vu)",
            "Web plate (d × tw) — FINAL", "Flange plate (bf × tf) — FINAL",
            "Web slenderness (d/tw) [Cl. 8.6.1.1]", "Shear slenderness λ_w [Cl. 8.4.2.2]",
            "Moment capacity (Md) [Cl. 8.2.1.2]", "Shear capacity (Vd) [Cl. 8.4.2.2]",
            "Moment utilization (Mu/Md)", "Shear utilization (Vu/Vd)",
            "Shear buckling method", "Deflection (δ) [Cl. 5.6.1]", "Deflection limit (L/300)",
            "Deflection check", "Intermediate stiffeners [Cl. 8.7.3]",
            "Inter. stiff. spacing", "Inter. stiff. size (bs×ts)",
            "Bearing stiffeners [Cl. 8.7.4]", "Bearing stiff. size (bb×tb) two flats",
            "Longitudinal stiffeners [Cl. 8.7.2] (if required)", "Steel grade",
            "Weight per meter", "Total weight"
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
            f"{stiff_spacing} mm c/c" if need_stiff else "—",
            f"{bs}×{ts} mm" if need_stiff else "—",
            f"Two flats {bb}×{tb} mm",
            f"{bb}×{tb} mm (outstand {bb}, thick {tb})",
            f"{bl}×{tl} mm at y={long_loc:.0f} mm" if (d/tw > 200*epsilon_val) else "Not required",
            steel_grade, f"{weight:.0f} kg/m", f"{total_weight:.0f} kg ({total_weight/1000:.2f} t)"
        ]
    }
    st.table(summary_data)
    
    st.success("✅ Design completed as per IS 800:2007.")
    
else:
    st.info("👈 Enter parameters and click **Design Plate Girder** to start.")
    st.markdown("""
    ### 📖 Design Steps (IS 800:2007 Clauses)
    1. Load combinations – **Cl. 5.3.3**  
    2. Required plastic modulus – **Cl. 8.2.1.2**  
    3. Economical depth – **Cl. 8.6**  
    4. Web thickness & shear capacity – **Cl. 8.4.2.2** & **Cl. 8.6.1.1**  
    5. Shear buckling resistance – **Cl. 8.4.2.2**  
    6. Web slenderness limits – **Cl. 8.6.1.1**  
    7. Flange compactness – **Cl. 8.6.2**  
    8. Deflection serviceability – **Cl. 5.6.1**  
    9. Intermediate stiffeners – **Cl. 8.7.3**  
    10. Bearing stiffeners – **Cl. 8.7.4**  
    11. Longitudinal stiffeners (optional) – **Cl. 8.7.2**  
    12. Auto‑revision to satisfy all criteria  
    """)
