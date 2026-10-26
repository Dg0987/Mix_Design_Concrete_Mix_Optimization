from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Concrete Mix Design Optimization API - IS 10262 / IS 456")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Models (once at startup) ──────────────────────────────────────────
try:
    if os.path.exists("xgboost_model.pkl"):
        model           = joblib.load("xgboost_model.pkl")
        company_to_code = joblib.load("company_to_code.pkl")
        type_to_code    = joblib.load("type_to_code.pkl")
        model_features  = joblib.load("model_features.pkl")
        print("Models loaded successfully.")
    else:
        print("Warning: xgboost_model.pkl not found. Run train_and_export.py first.")
except Exception as e:
    print(f"Error loading models: {e}")


# ─── IS 456 Table 5 — Exposure Class Limits ─────────────────────────────────
# (max_free_wc, min_cement_kg_m3, min_grade_fck)
IS456_TABLE5 = {
    "Mild"       : (0.55, 300, 20),
    "Moderate"   : (0.50, 300, 25),
    "Severe"     : (0.45, 320, 30),
    "Very Severe": (0.45, 340, 35),
    "Extreme"    : (0.40, 360, 40),
}


# ─── IS 10262 Cl. 3.3 — Standard Deviation for Target Mean Strength ─────────
def get_std_dev(fck: float) -> float:
    if fck <= 25:
        return 4.0
    elif fck <= 55:
        return 5.0
    else:
        return 6.0   # M60 and above


# ─── API Schemas ─────────────────────────────────────────────────────────────
class OptimizationRequest(BaseModel):
    fck: float
    exposure: str
    cement_company: str
    cement_type: str
    cement_grade: int


class MixResponse(BaseModel):
    water: float
    cement: float
    silica_fume: float
    fly_ash: float
    sand: float
    ca_10mm: float
    ca_20mm: float
    admixture: float
    wc_ratio: float
    wcm_ratio: float
    target_strength: float
    obtained_strength: float
    optimizer_success: bool
    units: dict


# ─── Health / Options Endpoints ───────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Concrete Mix Design Optimization API (IS 10262 / IS 456) is running!"}


@app.get("/options")
def get_options():
    return {
        "exposure_conditions": list(IS456_TABLE5.keys()),
        "cement_companies":    list(company_to_code.keys()),
        "cement_types":        list(type_to_code.keys()),
        "cement_grades":       [43, 53],
    }


# ─── Main Optimization Endpoint ──────────────────────────────────────────────
@app.post("/optimize", response_model=MixResponse)
def optimize_mix(request: OptimizationRequest):

    # ── Validate inputs ───────────────────────────────────────────────────
    if request.exposure not in IS456_TABLE5:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exposure. Options: {list(IS456_TABLE5.keys())}"
        )
    if request.cement_company not in company_to_code:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cement company. Options: {list(company_to_code.keys())}"
        )
    if request.cement_type not in type_to_code:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cement type. Options: {list(type_to_code.keys())}"
        )
    if not (15 <= request.fck <= 80):
        raise HTTPException(status_code=400, detail="fck must be between 15 and 80 MPa.")

    # ── IS Parameters ─────────────────────────────────────────────────────
    wc_max, cement_min, min_grade = IS456_TABLE5[request.exposure]

    # IS 10262 Cl. 3.3: fcm = fck + 1.65 * sigma
    target_strength = request.fck + 1.65 * get_std_dev(request.fck)

    # w/cm limit = same as w/c for IS 456 (IS 10262 uses w/cm for blended cements)
    wcm_max = wc_max

    c_comp_code = company_to_code[request.cement_company]
    c_type_code = type_to_code[request.cement_type]
    c_grade     = request.cement_grade

    # Column order MUST match training
    materials = ["Water", "Cement", "Silica_Fume", "Fly_Ash",
                 "Sand", "CA_10mm", "CA_20mm", "Admixture"]

    # ── Model Helper ──────────────────────────────────────────────────────
    def predict_strength(mix: np.ndarray) -> float:
        mix_df = pd.DataFrame([mix], columns=materials)
        cem_df = pd.DataFrame([{
            "Cement_Company": c_comp_code,
            "Cement_Type":    c_type_code,
            "Cement_Grade":   c_grade,
        }])
        full_input = pd.concat([mix_df, cem_df], axis=1)[model_features]
        return float(model.predict(full_input)[0])

    # ── Objective ─────────────────────────────────────────────────────────
    def objective(mix: np.ndarray) -> float:
        return (predict_strength(mix) - target_strength) ** 2

    # ── IS 10262 / IS 456 Constraints (all return >= 0 when satisfied) ────
    # mix[0]=Water  mix[1]=Cement   mix[2]=Silica_Fume  mix[3]=Fly_Ash
    # mix[4]=Sand   mix[5]=CA_10mm  mix[6]=CA_20mm      mix[7]=Admixture

    def c_wc_max(mix):
        """IS 456 Table 5: water/cement <= wc_max"""
        return wc_max * mix[1] - mix[0]

    def c_wcm_max(mix):
        """IS 456 / IS 10262: water/(cement+sf+fa) <= wcm_max"""
        return wcm_max * (mix[1] + mix[2] + mix[3]) - mix[0]

    def c_wc_min(mix):
        """IS 10262 Cl. 4.3: w/c >= 0.30 for adequate workability"""
        return mix[0] - 0.30 * mix[1]

    def c_cement_min(mix):
        """IS 456 Table 5: cement >= cement_min for exposure class"""
        return mix[1] - cement_min

    def c_cementitious_max(mix):
        """IS 10262 Cl. 5.2: total cementitious (cement+sf+fa) <= 450 kg/m3"""
        return 450.0 - (mix[1] + mix[2] + mix[3])

    def c_water_max(mix):
        """IS 10262 Table 2: water content <= 197 kg/m3"""
        return 197.0 - mix[0]

    def c_water_min(mix):
        """IS 10262: water >= 140 kg/m3 for adequate hydration"""
        return mix[0] - 140.0

    def c_sf_max(mix):
        """IS 15388: silica fume <= 10% of cement mass"""
        return 0.10 * mix[1] - mix[2]

    def c_fa_max(mix):
        """IS 1489 / IS 456: fly ash <= 35% of total cementitious"""
        return 0.35 * (mix[1] + mix[2] + mix[3]) - mix[3]

    def c_admix_max(mix):
        """IS 9103: admixture <= 2% of cement by mass"""
        return 0.020 * mix[1] - mix[7]

    def c_admix_min(mix):
        """IS 9103: admixture >= 0.5% of cement by mass"""
        return mix[7] - 0.005 * mix[1]

    def c_fine_agg_max(mix):
        """IS 10262 Cl. 4.4.2: sand/(sand+CA10+CA20) <= 0.45"""
        total_agg = mix[4] + mix[5] + mix[6]
        return 0.45 * total_agg - mix[4]

    def c_fine_agg_min(mix):
        """IS 10262 Cl. 4.4.2: sand/(sand+CA10+CA20) >= 0.30"""
        total_agg = mix[4] + mix[5] + mix[6]
        return mix[4] - 0.30 * total_agg

    def c_ca_ratio(mix):
        """IS 10262: CA_10mm >= 30% of coarse aggregate (CA10+CA20)"""
        return mix[5] - 0.30 * (mix[5] + mix[6])

    def c_total_upper(mix):
        """Fresh concrete unit weight <= 2450 kg/m3 (IS 10262 Annex A)"""
        return 2450.0 - sum(mix)

    def c_total_lower(mix):
        """Fresh concrete unit weight >= 2300 kg/m3"""
        return sum(mix) - 2300.0

    constraints = [
        {"type": "ineq", "fun": c_wc_max},
        {"type": "ineq", "fun": c_wcm_max},
        {"type": "ineq", "fun": c_wc_min},
        {"type": "ineq", "fun": c_cement_min},
        {"type": "ineq", "fun": c_cementitious_max},
        {"type": "ineq", "fun": c_water_max},
        {"type": "ineq", "fun": c_water_min},
        {"type": "ineq", "fun": c_sf_max},
        {"type": "ineq", "fun": c_fa_max},
        {"type": "ineq", "fun": c_admix_max},
        {"type": "ineq", "fun": c_admix_min},
        {"type": "ineq", "fun": c_fine_agg_max},
        {"type": "ineq", "fun": c_fine_agg_min},
        {"type": "ineq", "fun": c_ca_ratio},
        {"type": "ineq", "fun": c_total_upper},
        {"type": "ineq", "fun": c_total_lower},
    ]

    # ── Bounds (hard physical limits per IS 10262) ────────────────────────
    bounds = [
        (140, 200),           # Water   (kg/m3)
        (cement_min, 450),    # Cement  — lower = IS 456 Table 5 min for exposure
        (0, 60),              # Silica Fume (<=10% of max ~600 -> 60)
        (0, 160),             # Fly Ash (<=35% of 450 -> ~157)
        (450, 950),           # Sand
        (300, 750),           # CA_10mm
        (300, 850),           # CA_20mm
        (1.0, 12.0),          # Admixture (kg/m3)
    ]

    # ── Dynamic Initial Guess (IS 10262 Cl. 4 trial mix approach) ─────────
    # Cement scales with fck; water respects exposure w/c limit
    init_cement = float(np.clip(cement_min + (request.fck - 20) * 4.0, cement_min, 430))
    init_water  = float(np.clip(init_cement * wc_max * 0.88, 145, 190))
    init_sf     = float(np.clip(init_cement * 0.06, 0, 50))
    init_fa     = float(np.clip(init_cement * 0.12, 0, 130))
    total_cm    = init_cement + init_sf + init_fa
    remaining   = max(2380.0 - init_water - total_cm, 900.0)
    init_sand   = round(remaining * 0.38)
    init_ca10   = round(remaining * 0.28)
    init_ca20   = remaining - init_sand - init_ca10
    init_admix  = float(np.clip(init_cement * 0.012, 1.5, 10.0))
    x0 = [init_water, init_cement, init_sf, init_fa,
           init_sand,  init_ca10,   init_ca20, init_admix]

    # ── Primary: SLSQP (fast, gradient-based local optimizer) ────────────
    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
    )

    # ── Fallback: Differential Evolution (global, robust) ─────────────────
    # Triggers if SLSQP did not converge or produced constraint violations
    violation_tol = -1e-3
    slsqp_infeasible = (
        not result.success
        or any(c["fun"](result.x) < violation_tol for c in constraints)
    )
    if slsqp_infeasible:
        print(f"SLSQP fallback triggered (success={result.success}). Running DE...")

        def penalized(mix):
            val = objective(mix)
            penalty = sum(
                1e5 * (c["fun"](mix)) ** 2
                for c in constraints if c["fun"](mix) < 0
            )
            return val + penalty

        de_result = differential_evolution(
            penalized, bounds=bounds,
            seed=42, maxiter=500, popsize=15,
            tol=1e-8, mutation=(0.5, 1.5), recombination=0.9,
            polish=True,
        )
        if de_result.fun < result.fun:
            result = de_result

    optimal_mix = result.x

    # ── Final Prediction and Derived Ratios ───────────────────────────────
    pred_strength = predict_strength(optimal_mix)
    water   = optimal_mix[0]
    cement  = optimal_mix[1]
    sf      = optimal_mix[2]
    fa      = optimal_mix[3]
    total_cm = cement + sf + fa

    wc_ratio  = round(water / cement,   3) if cement   > 0 else 0.0
    wcm_ratio = round(water / total_cm, 3) if total_cm > 0 else 0.0

    return MixResponse(
        water             = round(float(water),          2),
        cement            = round(float(cement),         2),
        silica_fume       = round(float(sf),             2),
        fly_ash           = round(float(fa),             2),
        sand              = round(float(optimal_mix[4]), 2),
        ca_10mm           = round(float(optimal_mix[5]), 2),
        ca_20mm           = round(float(optimal_mix[6]), 2),
        admixture         = round(float(optimal_mix[7]), 2),
        wc_ratio          = wc_ratio,
        wcm_ratio         = wcm_ratio,
        target_strength   = round(float(target_strength), 2),
        obtained_strength = round(float(pred_strength),   2),
        optimizer_success = bool(result.success),
        units={
            "materials": "kg/m3",
            "strength":  "MPa",
            "wc_ratio":  "dimensionless",
        },
    )
