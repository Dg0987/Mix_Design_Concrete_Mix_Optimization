import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConcreteMix AI | IS 10262 Design Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0a0e1a,#0d1528,#0a1020); min-height:100vh; }
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.09);
}
/* fix all sidebar text to light */
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.07) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 8px !important;
}
.hero-banner {
    background: linear-gradient(135deg,rgba(56,189,248,.12),rgba(99,102,241,.12),rgba(168,85,247,.08));
    border: 1px solid rgba(56,189,248,.22);
    border-radius: 20px; padding:34px 40px; margin-bottom:24px; text-align:center;
}
.hero-title {
    font-size:2.4rem; font-weight:800;
    background:linear-gradient(135deg,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 8px;
}
.hero-sub { color:#94a3b8; font-size:1rem; margin:0; }
.sec-head {
    font-size:1.05rem; font-weight:700; color:#f1f5f9;
    border-left:3px solid #38bdf8; padding-left:12px; margin:22px 0 12px;
}
.kcard {
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,.09);
    border-radius:14px; padding:18px; text-align:center;
}
.kcard-blue {
    background:linear-gradient(135deg,rgba(56,189,248,.08),rgba(99,102,241,.05));
    border:1px solid rgba(56,189,248,.22); border-radius:14px; padding:18px; text-align:center;
}
.kcard-purple {
    background:linear-gradient(135deg,rgba(168,85,247,.08),rgba(244,114,182,.05));
    border:1px solid rgba(168,85,247,.22); border-radius:14px; padding:18px; text-align:center;
}
.row {
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 0; border-bottom:1px solid rgba(255,255,255,.06);
}
.row:last-child { border-bottom:none; }
.rlabel { color:#94a3b8; font-size:.87rem; }
.rval   { font-weight:700; font-size:.92rem; }
.is-tag {
    background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.3);
    color:#38bdf8; border-radius:5px; padding:1px 5px;
    font-size:.68rem; font-weight:600; margin-left:4px; vertical-align:middle;
}
.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(56,189,248,.4),transparent); margin:20px 0; }
.stButton>button {
    width:100%;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    padding:13px !important; font-size:.97rem !important; font-weight:700 !important;
    letter-spacing:.02em !important;
}
.stButton>button:hover { box-shadow:0 8px 24px rgba(99,102,241,.45) !important; }
.info-box {
    background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.2);
    border-left:3px solid #38bdf8; border-radius:8px;
    padding:9px 13px; color:#94a3b8; font-size:.83rem; margin:6px 0;
}
.col-head-blue {
    background:linear-gradient(135deg,rgba(56,189,248,.15),rgba(99,102,241,.07));
    border:1px solid rgba(56,189,248,.28); border-bottom:none;
    border-radius:12px 12px 0 0; padding:13px 18px;
    text-align:center; font-weight:700; font-size:.97rem; color:#38bdf8;
}
.col-head-purple {
    background:linear-gradient(135deg,rgba(168,85,247,.15),rgba(244,114,182,.07));
    border:1px solid rgba(168,85,247,.28); border-bottom:none;
    border-radius:12px 12px 0 0; padding:13px 18px;
    text-align:center; font-weight:700; font-size:.97rem; color:#c084fc;
}
.body-blue {
    background:linear-gradient(135deg,rgba(56,189,248,.05),rgba(99,102,241,.03));
    border:1px solid rgba(56,189,248,.22); border-radius:0 0 14px 14px; padding:18px;
}
.body-purple {
    background:linear-gradient(135deg,rgba(168,85,247,.05),rgba(244,114,182,.03));
    border:1px solid rgba(168,85,247,.22); border-radius:0 0 14px 14px; padding:18px;
}
.pill-ok { background:rgba(52,211,153,.15); border:1px solid rgba(52,211,153,.4);
    color:#34d399; border-radius:20px; padding:2px 11px; font-size:.78rem; font-weight:600; }
.pill-warn { background:rgba(251,191,36,.15); border:1px solid rgba(251,191,36,.4);
    color:#fbbf24; border-radius:20px; padding:2px 11px; font-size:.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  IS CODE CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════
IS456 = {
    "Mild"       : (0.55, 300, 20),
    "Moderate"   : (0.50, 300, 25),
    "Severe"     : (0.45, 320, 30),
    "Very Severe": (0.45, 340, 35),
    "Extreme"    : (0.40, 360, 40),
}
EXP_DESC = {
    "Mild"       : "Indoor / protected from weather",
    "Moderate"   : "Sheltered from rain, buried in soil",
    "Severe"     : "Alternating wet/dry, marine atmosphere",
    "Very Severe": "Sea water spray, corrosive fumes",
    "Extreme"    : "Abrasive / chemical attack conditions",
}

def sigma(fck):
    return 4.0 if fck <= 25 else (5.0 if fck <= 55 else 6.0)

def is_calc(fck, exp):
    wc_max, cem_min, _ = IS456[exp]
    sig   = sigma(fck)
    fcm   = round(fck + 1.65 * sig, 2)
    water = 186 if wc_max >= 0.55 else (180 if wc_max >= 0.50 else (172 if wc_max >= 0.45 else 165))
    wc    = min(wc_max, round(0.85 / (0.012 * fcm + 0.20), 3))
    cem   = max(round(water / wc, 1), cem_min)
    v_w   = water  / 1000.0
    v_c   = cem    / 3150.0
    v_agg = 1.0 - v_w - v_c - 0.02
    sand  = round(v_agg * 0.36 * 2650, 1)
    ca    = v_agg * 0.64 * 2680
    ca10  = round(ca * 0.40, 1)
    ca20  = round(ca * 0.60, 1)
    adm   = round(cem * 0.010, 2)
    return dict(
        fcm=fcm, sig=sig, wc=wc, wc_max=wc_max, cem_min=cem_min,
        water=round(water,1), cement=cem, sf=0.0, fa=0.0,
        sand=sand, ca10=ca10, ca20=ca20, adm=adm,
        total=round(water+cem+sand+ca10+ca20+adm, 1)
    )


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading ML model…")
def load_models():
    errors = []
    files  = {
        "xgboost_model.pkl"  : os.path.join(BASE, "xgboost_model.pkl"),
        "company_to_code.pkl": os.path.join(BASE, "company_to_code.pkl"),
        "type_to_code.pkl"   : os.path.join(BASE, "type_to_code.pkl"),
        "model_features.pkl" : os.path.join(BASE, "model_features.pkl"),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            errors.append(f"Missing: {name}  (looked in {BASE})")

    if errors:
        return None, {}, {}, [], errors

    try:
        mdl = joblib.load(files["xgboost_model.pkl"])
        c2c = joblib.load(files["company_to_code.pkl"])
        t2c = joblib.load(files["type_to_code.pkl"])
        mf  = joblib.load(files["model_features.pkl"])
        return mdl, c2c, t2c, mf, []
    except Exception as e:
        return None, {}, {}, [], [str(e)]

model, c2c, t2c, mfeats, load_errors = load_models()
MODEL_OK = (model is not None)
MATS     = ["Water","Cement","Silica_Fume","Fly_Ash","Sand","CA_10mm","CA_20mm","Admixture"]


# ═══════════════════════════════════════════════════════════════════
#  ML OPTIMIZER
# ═══════════════════════════════════════════════════════════════════
def ml_optimize(fck, exp, comp, ctype, cgrade):
    from scipy.optimize import minimize, differential_evolution

    wc_max, cem_min, _ = IS456[exp]
    target  = fck + 1.65 * sigma(fck)
    cc, ct  = c2c[comp], t2c[ctype]

    def predict(mix):
        row = dict(zip(MATS, mix))
        row.update({"Cement_Company": cc, "Cement_Type": ct, "Cement_Grade": cgrade})
        return float(model.predict(pd.DataFrame([row])[mfeats])[0])

    def obj(m): return (predict(m) - target) ** 2

    C = [
        {"type":"ineq","fun": lambda m: wc_max*(m[1])            - m[0]},
        {"type":"ineq","fun": lambda m: wc_max*(m[1]+m[2]+m[3])  - m[0]},
        {"type":"ineq","fun": lambda m: m[0] - 0.30*m[1]},
        {"type":"ineq","fun": lambda m: m[1] - cem_min},
        {"type":"ineq","fun": lambda m: 450  - (m[1]+m[2]+m[3])},
        {"type":"ineq","fun": lambda m: 197  - m[0]},
        {"type":"ineq","fun": lambda m: m[0] - 140},
        {"type":"ineq","fun": lambda m: 0.10*m[1] - m[2]},
        {"type":"ineq","fun": lambda m: 0.35*(m[1]+m[2]+m[3]) - m[3]},
        {"type":"ineq","fun": lambda m: 0.020*m[1] - m[7]},
        {"type":"ineq","fun": lambda m: m[7] - 0.005*m[1]},
        {"type":"ineq","fun": lambda m: 0.45*(m[4]+m[5]+m[6]) - m[4]},
        {"type":"ineq","fun": lambda m: m[4] - 0.30*(m[4]+m[5]+m[6])},
        {"type":"ineq","fun": lambda m: m[5] - 0.30*(m[5]+m[6])},
        {"type":"ineq","fun": lambda m: 2450 - sum(m)},
        {"type":"ineq","fun": lambda m: sum(m) - 2300},
    ]
    bounds = [(140,200),(cem_min,450),(0,60),(0,160),(450,950),(300,750),(300,850),(1,12)]

    ic  = float(np.clip(cem_min + (fck-20)*4, cem_min, 430))
    iw  = float(np.clip(ic*wc_max*0.88, 145, 190))
    isf = float(np.clip(ic*0.06, 0, 50))
    ifa = float(np.clip(ic*0.12, 0, 130))
    rem = max(2380 - iw - ic - isf - ifa, 900.0)
    isa = round(rem*0.38); ic10 = round(rem*0.28); ic20 = rem - isa - ic10
    iad = float(np.clip(ic*0.012, 1.5, 10))
    x0  = [iw, ic, isf, ifa, isa, ic10, ic20, iad]

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=C,
                   options={"maxiter":1000, "ftol":1e-9, "disp":False})

    if not res.success or any(c["fun"](res.x) < -1e-3 for c in C):
        def pen(m): return obj(m) + sum(1e5*c["fun"](m)**2 for c in C if c["fun"](m)<0)
        de = differential_evolution(pen, bounds, seed=42, maxiter=400,
                                    popsize=12, tol=1e-7, mutation=(0.5,1.5),
                                    recombination=0.9, polish=True)
        if de.fun < res.fun: res = de

    mx   = res.x
    pred = predict(mx)
    w, c, sf, fa = mx[0], mx[1], mx[2], mx[3]
    tcm  = c + sf + fa
    return dict(
        target=round(float(target),2), predicted=round(float(pred),2),
        wc =round(w/c,3) if c>0 else 0,
        wcm=round(w/tcm,3) if tcm>0 else 0,
        water=round(float(w),1),  cement=round(float(c),1),
        sf   =round(float(sf),1), fa    =round(float(fa),1),
        sand =round(float(mx[4]),1), ca10=round(float(mx[5]),1),
        ca20 =round(float(mx[6]),1), adm =round(float(mx[7]),2),
        total=round(float(sum(mx)),1), success=bool(res.success),
    )


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR  — always rendered
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:6px 0 18px;'>
        <div style='font-size:2rem;'>🏗️</div>
        <div style='font-size:.97rem;font-weight:700;color:#38bdf8;'>ConcreteMix AI</div>
        <div style='font-size:.70rem;color:#475569;margin-top:2px;'>IS 10262 / IS 456 Design Tool</div>
    </div>""", unsafe_allow_html=True)

    # ── Model status banner ──
    if MODEL_OK:
        st.success("✅ ML Model loaded successfully", icon=None)
    else:
        st.error("❌ **ML model not found!**")
        for e in load_errors:
            st.caption(e)
        st.info("Run `train_and_export.py` first, then restart this app.")
        st.markdown("---")
        st.markdown("**Working directory:** `" + BASE + "`")

    st.markdown('<div class="sec-head">📐 Design Parameters</div>', unsafe_allow_html=True)

    fck = st.number_input(
        "Characteristic Strength fck (MPa)",
        min_value=15.0, max_value=80.0, value=30.0, step=5.0,
        help="IS 456 — specify desired concrete grade (M15 to M80)"
    )
    st.markdown(
        f"<div class='info-box'>Grade: <b style='color:#38bdf8'>M{int(fck)}</b>"
        f"&nbsp;|&nbsp;σ = <b style='color:#38bdf8'>{sigma(fck)} MPa</b>"
        f"<span class='is-tag'>IS 10262 Cl.3.3</span></div>",
        unsafe_allow_html=True
    )

    exposure = st.selectbox(
        "Exposure Condition", list(IS456.keys()), index=1,
        help="IS 456 Table 5 — controls min cement & max w/c ratio"
    )
    st.markdown(f"<div class='info-box'>📍 {EXP_DESC[exposure]}</div>", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🏭 Cement Details</div>', unsafe_allow_html=True)

    companies = sorted(c2c.keys()) if MODEL_OK else ["(load model first)"]
    cement_company = st.selectbox("Cement Company", companies, disabled=not MODEL_OK)

    ctypes = sorted(t2c.keys()) if MODEL_OK else ["(load model first)"]
    cement_type = st.selectbox("Cement Type", ctypes, disabled=not MODEL_OK)

    cement_grade = st.radio("Cement Grade", [43, 53], horizontal=True)

    st.markdown("---")

    run_btn = st.button(
        "⚡  Run IS Code + ML Optimization",
        type="primary",
        disabled=not MODEL_OK,
        use_container_width=True,
    )

    if not MODEL_OK:
        if st.button("📐 Run IS Code Only (no ML)", use_container_width=True):
            st.session_state["run_is_only"] = True
    else:
        st.session_state["run_is_only"] = False

    st.markdown("""
    <div style='margin-top:16px;padding:11px;background:rgba(255,255,255,0.03);
         border:1px solid rgba(255,255,255,.07);border-radius:9px;
         font-size:.74rem;color:#475569;line-height:1.75;'>
        <b style='color:#64748b;'>IS Codes Referenced</b><br>
        • IS 456:2000 (Table 5)<br>• IS 10262:2019 (Cl.3.3, 4.3, 5.2)<br>
        • IS 9103 (Admixture) &nbsp;• IS 15388 (SF)<br>• IS 1489 (Fly Ash)
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🏗️ Concrete Mix Design AI</div>
    <div class='hero-sub'>IS 10262 / IS 456 Code Calculations &nbsp;⟷&nbsp; XGBoost ML Optimized Predictions</div>
</div>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
#  LANDING (nothing clicked yet)
# ────────────────────────────────────────────────────────────────────
is_only_mode = st.session_state.get("run_is_only", False)

if not run_btn and not is_only_mode:
    c1, c2, c3 = st.columns(3)
    for col, ic, clr, ttl, desc in zip(
        [c1,c2,c3],
        ["📐","🤖","📊"],
        ["#38bdf8","#c084fc","#34d399"],
        ["IS Code Method","ML Model Method","Side-by-Side Comparison"],
        ["IS 10262:2019 absolute volume method — target strength, w/c limits, aggregate proportioning.",
         "XGBoost trained on 200 real mix records, optimized via SLSQP + Differential Evolution.",
         "Interactive Plotly charts, IS compliance table, and one-click CSV export."],
    ):
        with col:
            st.markdown(f"""
            <div class='kcard' style='padding:26px 14px;'>
                <div style='font-size:2rem;margin-bottom:10px;'>{ic}</div>
                <div style='font-size:.92rem;font-weight:700;color:{clr};margin-bottom:8px;'>{ttl}</div>
                <div style='font-size:.80rem;color:#64748b;line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.warning(
            "**ML model files not found.** The button is disabled until the model is loaded.\n\n"
            f"Run `python train_and_export.py` inside `{BASE}`, then refresh this page.\n\n"
            "You can still click **Run IS Code Only** in the sidebar to see the IS 10262 calculation.",
            icon="⚠️"
        )
    else:
        st.markdown("""
        <div style='text-align:center;margin-top:32px;color:#334155;font-size:.88rem;'>
            👈 Set parameters in the sidebar then click <b style='color:#818cf8'>Run IS Code + ML Optimization</b>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ────────────────────────────────────────────────────────────────────
#  COMPUTE
# ────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Running calculations…"):
    ir = is_calc(fck, exposure)
    ml = ml_optimize(fck, exposure, cement_company, cement_type, cement_grade) if (run_btn and MODEL_OK) else None

if is_only_mode and not run_btn:
    ml = None

# ── Input summary chips ──
st.markdown('<div class="sec-head">🔢 Design Inputs</div>', unsafe_allow_html=True)
chip_cols = st.columns(5)
for col, lbl, val, clr in zip(
    chip_cols,
    ["Grade","Exposure","Company","Cement Type","C-Grade"],
    [f"M{int(fck)}", exposure, cement_company if MODEL_OK else "—",
     cement_type if MODEL_OK else "—", str(cement_grade)],
    ["#38bdf8","#c084fc","#34d399","#fbbf24","#f87171"],
):
    with col:
        st.markdown(f"""
        <div class='kcard' style='padding:11px;'>
            <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;'>{lbl}</div>
            <div style='font-size:1.05rem;font-weight:700;color:{clr};margin-top:2px;'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ════════ STRENGTH OVERVIEW ════════
st.markdown('<div class="sec-head">🎯 Strength Overview</div>', unsafe_allow_html=True)

sc = st.columns(4 if ml else 3)

with sc[0]:
    st.markdown(f"""
    <div class='kcard' style='text-align:center;padding:16px;'>
        <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;'>Characteristic fck</div>
        <div style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin:4px 0;'>{fck:.0f}&nbsp;<span style='font-size:.85rem;color:#64748b;'>MPa</span></div>
        <div style='font-size:.72rem;color:#475569;'>Design Input</div>
    </div>""", unsafe_allow_html=True)

with sc[1]:
    st.markdown(f"""
    <div class='kcard-blue' style='text-align:center;padding:16px;'>
        <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;'>IS Code — Target fcm</div>
        <div style='font-size:1.8rem;font-weight:800;color:#38bdf8;margin:4px 0;'>{ir['fcm']:.1f}&nbsp;<span style='font-size:.85rem;color:#64748b;'>MPa</span></div>
        <div style='font-size:.72rem;color:#475569;'>fck + 1.65σ &nbsp;<span class='is-tag'>IS 10262</span></div>
    </div>""", unsafe_allow_html=True)

with sc[2]:
    st.markdown(f"""
    <div class='kcard' style='text-align:center;padding:16px;'>
        <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;'>IS 456 Max W/C</div>
        <div style='font-size:1.8rem;font-weight:800;color:#fbbf24;margin:4px 0;'>{ir['wc_max']}</div>
        <div style='font-size:.72rem;color:#475569;'>Exposure: {exposure} &nbsp;<span class='is-tag'>Table 5</span></div>
    </div>""", unsafe_allow_html=True)

if ml and len(sc) > 3:
    diff  = ml['predicted'] - ml['target']
    dc    = "#34d399" if diff >= 0 else "#f87171"
    dl    = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
    with sc[3]:
        st.markdown(f"""
        <div class='kcard-purple' style='text-align:center;padding:16px;border-color:{dc}44;'>
            <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;'>ML — Predicted Strength</div>
            <div style='font-size:1.8rem;font-weight:800;color:{dc};margin:4px 0;'>{ml['predicted']:.1f}&nbsp;<span style='font-size:.85rem;color:#64748b;'>MPa</span></div>
            <div style='font-size:.72rem;color:{dc};'>{dl} MPa vs target</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ════════ MIX PROPORTIONS ════════
st.markdown('<div class="sec-head">📋 Mix Proportions Comparison</div>', unsafe_allow_html=True)

def row(label, val, color, tag=""):
    tg = f"<span class='is-tag'>{tag}</span>" if tag else ""
    return (f"<div class='row'><span class='rlabel'>{label}{tg}</span>"
            f"<span class='rval' style='color:{color};'>{val}</span></div>")

col_left, col_right = st.columns(2)

# ── IS Code column ──
with col_left:
    st.markdown('<div class="col-head-blue">📐 IS 10262 / IS 456 Code Method</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class='body-blue'>
        {row("Standard Deviation σ",       f"{ir['sig']} MPa",     "#38bdf8","Cl.3.3")}
        {row("Target Mean Strength",       f"{ir['fcm']} MPa",     "#38bdf8")}
        {row("Max W/C (IS 456 Table 5)",   str(ir['wc_max']),      "#38bdf8")}
        {row("Adopted W/C Ratio",          str(ir['wc']),          "#60a5fa")}
        {row("Min Cement Content",         f"{ir['cem_min']} kg/m³","#38bdf8","Table 5")}
        <div style='height:1px;background:rgba(56,189,248,.15);margin:8px 0;'></div>
        {row("💧 Water",    f"{ir['water']} kg/m³",  "#60a5fa")}
        {row("🏭 Cement",   f"{ir['cement']} kg/m³", "#38bdf8")}
        {row("🔬 Silica Fume",f"0.0 kg/m³",          "#818cf8")}
        {row("🌾 Fly Ash",  f"0.0 kg/m³",            "#818cf8")}
        {row("⬜ Sand",     f"{ir['sand']} kg/m³",   "#93c5fd")}
        {row("🪨 CA 10mm",  f"{ir['ca10']} kg/m³",   "#93c5fd")}
        {row("🪨 CA 20mm",  f"{ir['ca20']} kg/m³",   "#93c5fd")}
        {row("🧪 Admixture",f"{ir['adm']} kg/m³",    "#a5f3fc")}
        <div style='height:1px;background:rgba(56,189,248,.3);margin:8px 0;'></div>
        {row("📦 Total Unit Weight", f"{ir['total']} kg/m³","#38bdf8")}
    </div>""", unsafe_allow_html=True)

# ── ML column ──
with col_right:
    if ml:
        st.markdown('<div class="col-head-purple">🤖 XGBoost ML Optimized Method</div>', unsafe_allow_html=True)
        opt_pill = "<span class='pill-ok'>✅ Converged</span>" if ml['success'] else "<span class='pill-warn'>⚠️ Approx.</span>"
        diff  = ml['predicted'] - ml['target']
        dc    = "#34d399" if diff >= 0 else "#f87171"
        st.markdown(f"""
        <div class='body-purple'>
            <div class='row'><span class='rlabel'>Optimizer Status</span>{opt_pill}</div>
            {row("ML Predicted Strength",   f"{ml['predicted']} MPa", dc)}
            {row("Target Strength",         f"{ml['target']} MPa",    "#c084fc")}
            {row("W/C Ratio",               str(ml['wc']),            "#c084fc")}
            {row("W/CM Ratio",              str(ml['wcm']),           "#c084fc")}
            <div style='height:1px;background:rgba(168,85,247,.15);margin:8px 0;'></div>
            {row("💧 Water",     f"{ml['water']} kg/m³",  "#d8b4fe")}
            {row("🏭 Cement",    f"{ml['cement']} kg/m³", "#c084fc")}
            {row("🔬 Silica Fume",f"{ml['sf']} kg/m³",    "#a855f7")}
            {row("🌾 Fly Ash",   f"{ml['fa']} kg/m³",     "#a855f7")}
            {row("⬜ Sand",      f"{ml['sand']} kg/m³",   "#e879f9")}
            {row("🪨 CA 10mm",   f"{ml['ca10']} kg/m³",   "#e879f9")}
            {row("🪨 CA 20mm",   f"{ml['ca20']} kg/m³",   "#e879f9")}
            {row("🧪 Admixture", f"{ml['adm']} kg/m³",    "#f0abfc")}
            <div style='height:1px;background:rgba(168,85,247,.3);margin:8px 0;'></div>
            {row("📦 Total Unit Weight", f"{ml['total']} kg/m³","#c084fc")}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="col-head-purple">🤖 XGBoost ML Optimized Method</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='body-purple' style='text-align:center;padding:40px 20px;color:#64748b;'>
            <div style='font-size:1.6rem;margin-bottom:10px;'>🤖</div>
            ML model not available.<br>Run <code>train_and_export.py</code> first.
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ════════ CHARTS ════════
st.markdown('<div class="sec-head">📊 Visual Comparison</div>', unsafe_allow_html=True)

mat_names = ["Water","Cement","Silica Fume","Fly Ash","Sand","CA 10mm","CA 20mm","Admixture"]
is_vals = [ir['water'], ir['cement'], 0.0, 0.0, ir['sand'], ir['ca10'], ir['ca20'], ir['adm']]
ml_vals = ([ml['water'], ml['cement'], ml['sf'], ml['fa'],
            ml['sand'], ml['ca10'],  ml['ca20'], ml['adm']] if ml else None)

ch1, ch2 = st.columns(2)

PLOT_BG = "rgba(0,0,0,0)"
GRID    = "rgba(255,255,255,0.06)"

# Grouped bar
with ch1:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="IS Code", x=mat_names, y=is_vals, marker_color="#38bdf8",
                         opacity=0.85, text=[f"{v:.0f}" for v in is_vals],
                         textposition="outside", textfont=dict(color="#e2e8f0",size=10)))
    if ml_vals:
        fig.add_trace(go.Bar(name="ML Optimized", x=mat_names, y=ml_vals, marker_color="#c084fc",
                             opacity=0.85, text=[f"{v:.0f}" for v in ml_vals],
                             textposition="outside", textfont=dict(color="#e2e8f0",size=10)))
    fig.update_layout(
        title=dict(text="Material Quantities (kg/m³)", font=dict(color="#94a3b8",size=13)),
        barmode="group", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(color="#94a3b8"), height=360,
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID, title="kg/m³"),
        legend=dict(bgcolor=PLOT_BG, font=dict(color="#94a3b8")),
        margin=dict(t=45,b=10,l=10,r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# Donut(s)
with ch2:
    COLORS_IS = ["#0ea5e9","#3b82f6","#8b5cf6","#a78bfa","#f59e0b","#d97706","#b45309","#0d9488"]
    COLORS_ML = ["#c084fc","#a855f7","#7c3aed","#db2777","#f59e0b","#d97706","#b45309","#0d9488"]
    fig2 = go.Figure()
    if ml_vals:
        fig2.add_trace(go.Pie(values=is_vals, labels=mat_names, hole=0.52,
                              domain={"x":[0,0.46]}, marker_colors=COLORS_IS,
                              textinfo="percent", textfont=dict(color="#f1f5f9",size=9),
                              hovertemplate="%{label}: %{value:.0f} kg/m³<extra></extra>"))
        fig2.add_trace(go.Pie(values=ml_vals, labels=mat_names, hole=0.52,
                              domain={"x":[0.54,1.0]}, marker_colors=COLORS_ML,
                              textinfo="percent", textfont=dict(color="#f1f5f9",size=9),
                              hovertemplate="%{label}: %{value:.0f} kg/m³<extra></extra>"))
        annotations = [
            dict(text="IS Code",x=0.19,y=0.5,font_size=11,showarrow=False,font_color="#38bdf8"),
            dict(text="ML",     x=0.81,y=0.5,font_size=11,showarrow=False,font_color="#c084fc"),
        ]
    else:
        fig2.add_trace(go.Pie(values=is_vals, labels=mat_names, hole=0.52,
                              marker_colors=COLORS_IS, textinfo="percent",
                              textfont=dict(color="#f1f5f9",size=10),
                              hovertemplate="%{label}: %{value:.0f} kg/m³<extra></extra>"))
        annotations = [dict(text="IS Code",x=0.5,y=0.5,font_size=12,showarrow=False,font_color="#38bdf8")]
    fig2.update_layout(
        title=dict(text="Mix Proportion Share", font=dict(color="#94a3b8",size=13)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, showlegend=False,
        font=dict(color="#94a3b8"), height=360, margin=dict(t=45,b=10,l=10,r=10),
        annotations=annotations,
    )
    st.plotly_chart(fig2, use_container_width=True)

# Radar
st.markdown('<div class="sec-head">🕸️ Radar — Material Balance</div>', unsafe_allow_html=True)
rl    = ["Water","Cement","Sand","CA 10mm","CA 20mm","Admixture ×10"]
ir_r  = [ir['water'],ir['cement'],ir['sand'],ir['ca10'],ir['ca20'],ir['adm']*10]
fig3  = go.Figure()
fig3.add_trace(go.Scatterpolar(r=ir_r+[ir_r[0]], theta=rl+[rl[0]], fill="toself",
                                name="IS Code", line=dict(color="#38bdf8",width=2),
                                fillcolor="rgba(56,189,248,0.09)"))
if ml:
    ml_r = [ml['water'],ml['cement'],ml['sand'],ml['ca10'],ml['ca20'],ml['adm']*10]
    fig3.add_trace(go.Scatterpolar(r=ml_r+[ml_r[0]], theta=rl+[rl[0]], fill="toself",
                                    name="ML Optimized", line=dict(color="#c084fc",width=2),
                                    fillcolor="rgba(192,132,252,0.09)"))
fig3.update_layout(
    polar=dict(bgcolor=PLOT_BG,
               radialaxis=dict(visible=True,gridcolor="rgba(255,255,255,0.07)",color="#475569"),
               angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",color="#94a3b8")),
    plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
    legend=dict(bgcolor=PLOT_BG,font=dict(color="#94a3b8")),
    font=dict(color="#94a3b8"), height=400, margin=dict(t=20,b=20,l=60,r=60),
)
st.plotly_chart(fig3, use_container_width=True)


# ════════ IS CODE COMPLIANCE TABLE (ML only) ════════
if ml:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">✅ IS Code Compliance — ML Result</div>', unsafe_allow_html=True)
    wc_lim, cem_lim, _ = IS456[exposure]
    checks = [
        (f"W/C ≤ {wc_lim}",           ml['wc']        <= wc_lim,      f"{ml['wc']}",          "IS 456 Table 5"),
        (f"W/CM ≤ {wc_lim}",          ml['wcm']       <= wc_lim,      f"{ml['wcm']}",         "IS 10262"),
        (f"Cement ≥ {cem_lim} kg/m³", ml['cement']    >= cem_lim,     f"{ml['cement']} kg/m³","IS 456 Table 5"),
        ("Water ≤ 197 kg/m³",          ml['water']     <= 197,         f"{ml['water']} kg/m³", "IS 10262 Table 2"),
        ("Total ≤ 2450 kg/m³",         ml['total']     <= 2450,        f"{ml['total']} kg/m³", "IS 10262 Annex A"),
        ("Total ≥ 2300 kg/m³",         ml['total']     >= 2300,        f"{ml['total']} kg/m³", "IS 10262 Annex A"),
        ("Predicted ≥ 97% of Target",  ml['predicted'] >= ml['target']*0.97,
                                                                        f"{ml['predicted']} MPa","IS 10262 Cl.3.3"),
    ]
    rows = [{"Criterion":c, "Adopted Value":v, "Status":"✅ Pass" if p else "❌ Fail", "IS Clause":cl}
            for c,p,v,cl in checks]
    df_c = pd.DataFrame(rows)
    st.dataframe(
        df_c.style.map(
            lambda v: ("color:#34d399;font-weight:600" if "✅" in str(v)
                       else ("color:#f87171;font-weight:600" if "❌" in str(v) else "")),
            subset=["Status"]
        ),
        use_container_width=True, hide_index=True,
    )


# ════════ EXPORT ════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-head">💾 Export Results</div>', unsafe_allow_html=True)

export = {
    "Parameter": ["Water","Cement","Silica Fume","Fly Ash","Sand","CA 10mm","CA 20mm",
                  "Admixture","Total","W/C Ratio","W/CM Ratio","Target fcm (MPa)","Predicted (MPa)"],
    "Unit":      ["kg/m³"]*9 + ["—","—","MPa","MPa"],
    "IS Code":   [ir['water'],ir['cement'],0,0,ir['sand'],ir['ca10'],ir['ca20'],ir['adm'],
                  ir['total'],ir['wc'],ir['wc'],"—",ir['fcm']],
}
if ml:
    export["ML Optimized"] = [ml['water'],ml['cement'],ml['sf'],ml['fa'],ml['sand'],
                               ml['ca10'],ml['ca20'],ml['adm'],ml['total'],
                               ml['wc'],ml['wcm'],ml['target'],ml['predicted']]
else:
    export["ML Optimized"] = ["—"] * 13

df_exp = pd.DataFrame(export)
dl_col, tbl_col = st.columns([1, 3])
with dl_col:
    csv = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv,
                       file_name=f"mix_M{int(fck)}_{exposure.replace(' ','_')}.csv",
                       mime="text/csv", use_container_width=True)
    st.markdown(f"""
    <div style='margin-top:12px;padding:10px;background:rgba(255,255,255,.03);
         border:1px solid rgba(255,255,255,.07);border-radius:8px;
         font-size:.75rem;color:#475569;'>
        <b>Grade:</b> M{int(fck)}<br>
        <b>Exposure:</b> {exposure}<br>
        <b>Target fcm:</b> {ir['fcm']} MPa<br>
        {"<b>ML Predicted:</b> "+str(ml['predicted'])+" MPa" if ml else "ML: not run"}
    </div>""", unsafe_allow_html=True)
with tbl_col:
    st.dataframe(df_exp, use_container_width=True, hide_index=True)

st.markdown("""
<div style='text-align:center;padding:24px 0 8px;color:#1e293b;font-size:.76rem;'>
    ConcreteMix AI &nbsp;|&nbsp; IS 10262:2019 &amp; IS 456:2000 &nbsp;|&nbsp; XGBoost ML Optimizer
</div>""", unsafe_allow_html=True)
