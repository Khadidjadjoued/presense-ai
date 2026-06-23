import streamlit as st
import streamlit.components.v1 as components
import joblib, json
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
import os

RANDOM_STATE = 42
TOP_K = 20

# ── Base directory: works locally AND on Streamlit Cloud ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class CombinedSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=TOP_K):
        self.k = k
        self.selected_indices_ = None
        self._n_features_in    = None
    def fit(self, X, y):
        self._n_features_in = X.shape[1]
        scores_f  = f_classif(X, y)[0]
        scores_mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        def norm(s):
            r = s - s.min()
            return r / (r.max() + 1e-10)
        combined = norm(scores_f) + norm(scores_mi)
        self.selected_indices_ = np.argsort(combined)[::-1][:self.k]
        return self
    def transform(self, X):
        self._n_features_in = X.shape[1]
        return X[:, self.selected_indices_]
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
    def get_support(self):
        mask = np.zeros(self._n_features_in, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

SELECTED_BIOMARKERS = [
    {"sl": "SL000268", "name": "Angiostatin",                             "median": 19847.95},
    {"sl": "SL000525", "name": "Matrilysin",                              "median": 737.05},
    {"sl": "SL000584", "name": "Transforming growth factor beta-1",       "median": 2502.4},
    {"sl": "SL003994", "name": "Bone morphogenetic protein 1",            "median": 2057.85},
    {"sl": "SL008380", "name": "Cathepsin Z",                             "median": 4172.4},
    {"sl": "SL004643", "name": "Angiopoietin-4",                          "median": 550.05},
    {"sl": "SL002662", "name": "Coagulation Factor XI",                   "median": 1764.25},
    {"sl": "SL007373", "name": "Peptidyl-prolyl cis-trans isomerase D",   "median": 4379.4},
    {"sl": "SL003643", "name": "Glutathione S-transferase P",             "median": 11400.55},
    {"sl": "SL007471", "name": "Collectin-12",                            "median": 2014.8},
    {"sl": "SL011202", "name": "Alpha-soluble NSF attachment protein",    "median": 10683.4},
    {"sl": "SL003524", "name": "Protein disulfide-isomerase A3",          "median": 15251.0},
    {"sl": "SL007640", "name": "C-type lectin domain family 7A",          "median": 1983.1},
    {"sl": "SL001737", "name": "14-3-3 protein sigma",                    "median": 1019.15},
    {"sl": "SL008933", "name": "Protein DJ-1",                            "median": 556.75},
    {"sl": "SL010510", "name": "Serine/threonine-protein kinase PAK 7",   "median": 899.75},
    {"sl": "SL002803", "name": "Ubiquitin carboxyl-terminal hydrolase L1","median": 3206.2},
    {"sl": "SL003328", "name": "Complement factor I",                     "median": 58560.2},
    {"sl": "SL008822", "name": "EMR2 (EGF-like receptor)",                "median": 454.0},
    {"sl": "SL005217", "name": "Siglec-6",                                "median": 1072.35},
]

st.set_page_config(page_title="PreSense AI", page_icon="🤱",
                   layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def load_assets():
    lr  = joblib.load(os.path.join(BASE_DIR, "lr_model.pkl"))
    rf  = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
    with open(os.path.join(BASE_DIR, "model_info.json")) as f:
        info = json.load(f)
    return lr, rf, xgb, info

lr_model, rf_model, xgb_model, model_info = load_assets()

MODELS = {"Logistic Regression": lr_model,
          "Random Forest":       rf_model,
          "XGBoost":             xgb_model}

MODEL_STATS = {
    "Logistic Regression": {"AUC":"0.747","Sens":"93.8%","Spec":"54.5%","F1":"0.732","Acc":"71.1%","thr":0.272},
    "Random Forest":       {"AUC":"0.705","Sens":"100%", "Spec":"45.5%","F1":"0.727","Acc":"68.4%","thr":0.314},
    "XGBoost":             {"AUC":"0.739","Sens":"100%", "Spec":"40.9%","F1":"0.711","Acc":"65.8%","thr":0.205},
}

for k,v in [("page","home"),("pred_proba",None),("pred_risk","High Risk"),
            ("pred_model","Logistic Regression"),("patient_data",{}),
            ("shap_values",None),("shap_features",None),("shap_base",None),
            ("shap_inputs",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

def compute_patient_shap(pipe, X_input_df, model_name, feature_cols):
    preprocessor = pipe["preprocessor"]
    clf          = pipe["clf"]
    protein_pipe  = preprocessor.named_transformers_["protein"]
    var_filter    = protein_pipe.named_steps["var_filter"]
    selector      = protein_pipe.named_steps["selector"]
    protein_cols  = model_info["protein_cols"]
    all_clinical  = model_info["all_clinical"]
    cols_after_var = np.array(protein_cols)[var_filter.get_support()]
    cols_selected  = cols_after_var[selector.selected_indices_]
    readable_names = list(all_clinical) + list(cols_selected)
    X_tr = preprocessor.transform(X_input_df)
    X_df = pd.DataFrame(X_tr, columns=readable_names)
    if model_name == "Logistic Regression":
        bg = pd.DataFrame(np.zeros((1, len(readable_names))), columns=readable_names)
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(X_df)
        if isinstance(sv, list): sv = sv[1]
        base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[1])
    else:
        explainer = shap.TreeExplainer(clf)
        sv_raw = explainer.shap_values(X_df)
        if isinstance(sv_raw, list): sv = sv_raw[1]
        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3: sv = sv_raw[:, :, 1]
        else: sv = sv_raw
        ev = explainer.expected_value
        base_val = float(ev[1]) if hasattr(ev, '__len__') else float(ev)
    shap_vals = sv[0]
    feat_names = readable_names
    idx_sorted = np.argsort(np.abs(shap_vals))[::-1][:15]
    top_names  = [feat_names[i] for i in idx_sorted]
    top_vals   = [float(shap_vals[i]) for i in idx_sorted]
    top_inputs = [float(X_df.iloc[0, i]) for i in idx_sorted]
    return top_names, top_vals, top_inputs, base_val

params = st.query_params
if params.get("action") == "predict":
    try:
        age       = float(params.get("age",       25.0))
        bmi       = float(params.get("bmi",       27.4))
        gravidity = float(params.get("gravidity", 2.0))
        parity    = float(params.get("parity",    1.0))
        ga        = float(params.get("ga",        11.7))
        sel_model = params.get("model", "Logistic Regression")
        feature_cols = model_info["feature_cols"]
        input_dict   = {col: np.nan for col in feature_cols}
        input_dict["GA"]        = ga
        input_dict["Age"]       = age
        input_dict["BMI"]       = bmi
        input_dict["Gravidity"] = gravidity
        input_dict["Parity"]    = parity
        for bio in SELECTED_BIOMARKERS:
            val = params.get(bio["sl"])
            if val is not None:
                try: input_dict[bio["sl"]] = float(val)
                except: pass
        X_input = pd.DataFrame([input_dict])[feature_cols]
        model   = MODELS[sel_model]
        proba   = float(model.predict_proba(X_input)[0][1])
        thr     = MODEL_STATS[sel_model]["thr"]
        risk    = "High Risk" if proba >= thr else "Low Risk"
        n_provided = sum(1 for bio in SELECTED_BIOMARKERS if params.get(bio["sl"]) is not None)
        try:
            top_names, top_vals, top_inputs, base_val = compute_patient_shap(model, X_input, sel_model, feature_cols)
            st.session_state.shap_values   = top_vals
            st.session_state.shap_features = top_names
            st.session_state.shap_inputs   = top_inputs
            st.session_state.shap_base     = base_val
        except:
            st.session_state.shap_values   = None
            st.session_state.shap_features = None
            st.session_state.shap_inputs   = None
            st.session_state.shap_base     = None
        st.session_state.pred_proba   = proba
        st.session_state.pred_risk    = risk
        st.session_state.pred_model   = sel_model
        st.session_state.patient_data = {"Age": age, "BMI": bmi, "Gravidity": gravidity,
            "Parity": parity, "GA": ga, "n_biomarkers": n_provided, "missing": len(SELECTED_BIOMARKERS) - n_provided}
        st.session_state.page = "results"
        st.query_params.clear()
        st.rerun()
    except Exception as ex:
        st.error(f"Prediction error: {ex}")

st.markdown("""
<style>
header,#MainMenu,footer,[data-testid="stToolbar"],
[data-testid="stSidebar"],[data-testid="collapsedControl"],
[data-testid="stHorizontalBlock"]{display:none!important}
.stApp,[data-testid="stMain"]{background:#f0f4fb!important}
.block-container{padding:0!important;max-width:100%!important}
[data-testid="stVerticalBlock"]{gap:0!important;padding:0!important}
</style>""", unsafe_allow_html=True)

b = st.columns(5)
keys = ["home","patient","results","shap","perf"]
for i in range(5):
    if b[i].button("x", key=keys[i]):
        st.session_state.page = keys[i]
        st.rerun()

cur = st.session_state.page

FONTS = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>'

BASE_CSS = """<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --blue:#1847b5;--blue2:#2563eb;--blue-light:#3b82f6;
  --soft:#eff6ff;--soft2:#dbeafe;
  --bg:#f0f4fb;--w:#ffffff;
  --t1:#0f172a;--t2:#334155;--t3:#64748b;--t4:#94a3b8;
  --bd:#e2e8f0;--bd2:#cbd5e1;
  --green:#059669;--red:#dc2626;--amber:#d97706;--purple:#7c3aed;
  --s1:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  --s2:0 4px 6px -1px rgba(15,23,42,.07),0 2px 4px -1px rgba(15,23,42,.04);
  --s3:0 10px 15px -3px rgba(15,23,42,.08),0 4px 6px -2px rgba(15,23,42,.04);
  --s4:0 20px 25px -5px rgba(24,71,181,.12),0 10px 10px -5px rgba(24,71,181,.06);
  --sb:0 4px 14px rgba(24,71,181,.35);
  --tf:all .2s cubic-bezier(.4,0,.2,1);
  --r:16px;--r2:12px;--r3:8px;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--t1);display:flex;min-height:100vh}

/* ── Sidebar ── */
.sb{
  width:220px;min-width:220px;background:var(--w);
  border-right:1px solid var(--bd);
  display:flex;flex-direction:column;padding:24px 12px 20px;
  position:fixed;top:0;left:0;bottom:0;z-index:100;
  box-shadow:2px 0 16px rgba(15,23,42,.06)
}
.logo-wrap{display:flex;align-items:center;gap:12px;padding:0 8px;margin-bottom:28px}
.logo-icon{width:44px;height:44px;border-radius:12px;
  background:linear-gradient(135deg,#1847b5,#3b82f6);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:var(--sb)}
.logo-icon svg{width:24px;height:24px}
.logo-text .ln{font-size:15px;font-weight:800;color:var(--blue);letter-spacing:-.3px;line-height:1.1}
.logo-text .ls{font-size:10px;color:var(--t4);line-height:1.4;margin-top:2px}

nav{display:flex;flex-direction:column;gap:2px;margin-bottom:auto}
.nav-label{font-size:10px;font-weight:600;color:var(--t4);text-transform:uppercase;
  letter-spacing:.8px;padding:0 10px;margin-bottom:6px;margin-top:4px}
.ni{
  display:flex;align-items:center;gap:10px;padding:9px 10px;
  border-radius:var(--r3);font-size:13px;font-weight:500;color:var(--t3);
  cursor:pointer;transition:var(--tf);border:none;background:transparent;
  width:100%;text-align:left;position:relative
}
.ni:hover{background:#f1f5f9;color:var(--t1)}
.ni.on{background:var(--soft);color:var(--blue);font-weight:600}
.ni .ni-dot{
  width:6px;height:6px;border-radius:50%;background:var(--blue);
  position:absolute;right:10px;opacity:0;transition:var(--tf)
}
.ni.on .ni-dot{opacity:1}
.ni-icon{width:32px;height:32px;border-radius:var(--r3);
  display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ni.on .ni-icon{background:var(--soft2)}
.ni img{width:16px;height:16px;object-fit:contain}

.sb-footer{
  background:linear-gradient(135deg,#eff6ff,#e8f0fe);
  border:1px solid #c7d7fa;border-radius:var(--r2);
  padding:13px 12px;margin-top:16px;
  display:flex;align-items:flex-start;gap:9px
}
.sb-footer-title{font-size:11px;font-weight:700;color:var(--blue);line-height:1.5}
.ecg{margin-top:12px;opacity:.18}
.ecg svg{width:100%}

/* ── Main area ── */
.main{margin-left:220px;flex:1;min-height:100vh;background:var(--bg)}

/* ── Top header bar ── */
.topbar{
  background:var(--w);border-bottom:1px solid var(--bd);
  padding:0 36px;height:60px;display:flex;align-items:center;
  justify-content:space-between;position:sticky;top:0;z-index:50;
  box-shadow:var(--s1)
}
.topbar-left{display:flex;flex-direction:column}
.topbar-title{font-size:20px;font-weight:800;color:var(--t1);letter-spacing:-.4px;line-height:1.1}
.topbar-sub{font-size:11.5px;color:var(--t3);margin-top:2px}
.topbar-right{display:flex;align-items:center;gap:10px}
.tb-badge{
  display:inline-flex;align-items:center;gap:6px;
  background:#f8fafc;border:1px solid var(--bd);
  border-radius:999px;padding:5px 12px;
  font-size:11px;font-weight:500;color:var(--t2)
}
.tb-badge .dot{width:6px;height:6px;border-radius:50%;background:#10b981;flex-shrink:0}
.tb-help{
  width:30px;height:30px;border-radius:50%;
  background:#f1f5f9;border:1px solid var(--bd);
  display:flex;align-items:center;justify-content:center;cursor:pointer
}
.tb-help img{width:15px;height:15px;opacity:.5}

/* ── Page content ── */
.content{padding:32px 36px 40px}

/* ── Hero banner ── */
.hero{
  background:linear-gradient(135deg,var(--blue) 0%,#2563eb 60%,#3b82f6 100%);
  border-radius:var(--r);padding:36px 40px;
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:28px;position:relative;overflow:hidden;
  box-shadow:var(--s4)
}
.hero::before{
  content:'';position:absolute;top:-60px;right:180px;
  width:300px;height:300px;border-radius:50%;
  background:rgba(255,255,255,.06);pointer-events:none
}
.hero::after{
  content:'';position:absolute;bottom:-80px;right:-40px;
  width:280px;height:280px;border-radius:50%;
  background:rgba(255,255,255,.05);pointer-events:none
}
.hero-text{position:relative;z-index:1;max-width:520px}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);
  border-radius:999px;padding:4px 12px;
  font-size:10.5px;font-weight:600;color:rgba(255,255,255,.9);
  letter-spacing:.3px;margin-bottom:14px
}
.hero-h{font-size:36px;font-weight:800;color:#fff;letter-spacing:-.8px;line-height:1.1;margin-bottom:12px}
.hero-p{font-size:13.5px;color:rgba(255,255,255,.8);line-height:1.75;margin-bottom:22px;max-width:420px}
.hero-btn{
  display:inline-flex;align-items:center;gap:9px;
  background:#fff;color:var(--blue);
  border:none;border-radius:var(--r2);padding:11px 22px;
  font-size:13.5px;font-weight:700;cursor:pointer;
  box-shadow:0 4px 12px rgba(0,0,0,.15);transition:var(--tf)
}
.hero-btn:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.2)}
.hero-btn img{width:16px;height:16px}
.hero-visual{
  position:relative;z-index:1;flex-shrink:0;
  width:320px;height:260px;display:flex;align-items:center;justify-content:center
}
.hero-img-ring{
  width:240px;height:240px;border-radius:50%;
  border:3px solid rgba(255,255,255,.25);
  display:flex;align-items:center;justify-content:center
}
.hero-img-inner{
  width:220px;height:220px;border-radius:50%;
  background:rgba(255,255,255,.12);
  display:flex;align-items:center;justify-content:center;
  overflow:hidden
}
.hero-img-inner img{
  width:210px;height:210px;border-radius:50%;
  object-fit:cover;object-position:top;
  border:4px solid rgba(255,255,255,.3)
}
/* floating stats on hero */
.hero-stat{
  position:absolute;background:rgba(255,255,255,.95);
  border-radius:var(--r2);padding:9px 14px;
  box-shadow:0 8px 24px rgba(0,0,0,.12);backdrop-filter:blur(8px)
}
.hero-stat.top-left{top:12px;left:0}
.hero-stat.bottom-right{bottom:8px;right:0}
.hero-stat-val{font-size:18px;font-weight:800;color:var(--blue);line-height:1}
.hero-stat-lbl{font-size:10px;color:var(--t3);margin-top:2px}

/* ── Feature cards ── */
.feat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.feat-card{
  background:var(--w);border-radius:var(--r);border:1px solid var(--bd);
  padding:22px 18px;box-shadow:var(--s1);transition:var(--tf);
  position:relative;overflow:hidden
}
.feat-card::after{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  opacity:0;transition:var(--tf)
}
.feat-card:hover{transform:translateY(-3px);box-shadow:var(--s3);border-color:var(--bd2)}
.feat-card:hover::after{opacity:1}
.fc-blue::after{background:linear-gradient(90deg,#1847b5,#60a5fa)}
.fc-green::after{background:linear-gradient(90deg,#059669,#34d399)}
.fc-purple::after{background:linear-gradient(90deg,#7c3aed,#a78bfa)}
.fc-pink::after{background:linear-gradient(90deg,#db2777,#f9a8d4)}
.feat-icon{
  width:44px;height:44px;border-radius:var(--r2);
  display:flex;align-items:center;justify-content:center;
  margin-bottom:14px;box-shadow:var(--s1)
}
.feat-icon img{width:22px;height:22px}
.fi-blue{background:linear-gradient(135deg,#eff6ff,#dbeafe)}
.fi-green{background:linear-gradient(135deg,#ecfdf5,#d1fae5)}
.fi-purple{background:linear-gradient(135deg,#f5f3ff,#ede9fe)}
.fi-pink{background:linear-gradient(135deg,#fdf2f8,#fce7f3)}
.feat-title{font-size:13.5px;font-weight:700;color:var(--t1);margin-bottom:5px}
.feat-desc{font-size:12px;color:var(--t3);line-height:1.65}

/* ── Generic card ── */
.card{background:var(--w);border-radius:var(--r);border:1px solid var(--bd);box-shadow:var(--s1)}
.card-header{
  display:flex;align-items:center;gap:10px;
  padding:16px 20px;border-bottom:1px solid var(--bd)
}
.card-icon{
  width:34px;height:34px;border-radius:var(--r3);
  display:flex;align-items:center;justify-content:center;flex-shrink:0
}
.card-icon img{width:17px;height:17px}
.card-title{font-size:14px;font-weight:700;color:var(--t1)}
.card-sub{font-size:11.5px;color:var(--t3);margin-top:1px}
.card-body{padding:20px}

/* ── Form inputs ── */
.form-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px}
.field{margin-bottom:14px}
.field-label{font-size:12px;font-weight:500;color:var(--t2);margin-bottom:5px;display:block}
.input-wrap{
  display:flex;align-items:center;
  background:#f8fafc;border:1.5px solid var(--bd);
  border-radius:var(--r3);overflow:hidden;transition:border-color .18s
}
.input-wrap:focus-within{border-color:var(--blue);background:#fff}
.input-wrap input{
  flex:1;border:none;background:transparent;
  padding:9px 12px;font-size:13px;font-family:'Inter',sans-serif;
  color:var(--t1);outline:none;min-width:0
}
.input-unit{
  padding:9px 10px;font-size:11px;color:var(--t3);font-weight:500;
  border-left:1px solid var(--bd);background:#f1f5f9;white-space:nowrap
}
.select-wrap select{
  width:100%;background:#f8fafc;border:1.5px solid var(--bd);
  border-radius:var(--r3);padding:9px 12px;
  font-size:13px;font-family:'Inter',sans-serif;
  color:var(--t1);outline:none;cursor:pointer;transition:border-color .18s;
  appearance:none;-webkit-appearance:none
}
.select-wrap select:focus{border-color:var(--blue);background:#fff}
.select-arrow{position:relative}
.select-arrow::after{
  content:'▾';position:absolute;right:12px;top:50%;
  transform:translateY(-50%);font-size:12px;color:var(--t3);pointer-events:none
}

/* ── Biomarker row ── */
.bio-row{
  display:flex;align-items:center;gap:8px;
  padding:7px 10px;border-radius:var(--r3);transition:background .15s
}
.bio-row:hover{background:#f8fafc}
.bio-name{flex:1;font-size:11.5px;color:var(--t2);font-weight:500;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bio-input-wrap{
  display:flex;align-items:center;
  background:#f8fafc;border:1.5px solid var(--bd);
  border-radius:var(--r3);overflow:hidden;transition:border-color .18s;
  min-width:140px
}
.bio-input-wrap:focus-within{border-color:var(--blue);background:#fff}
.bio-input-wrap input{
  flex:1;border:none;background:transparent;
  padding:6px 8px;font-size:11.5px;font-family:'Inter',sans-serif;
  color:var(--t1);outline:none;width:60px
}
.bio-unit{
  padding:6px 7px;font-size:10px;color:var(--t3);font-weight:500;
  border-left:1px solid var(--bd);background:#f1f5f9
}

/* ── Buttons ── */
.btn-primary{
  display:inline-flex;align-items:center;justify-content:center;gap:8px;
  background:linear-gradient(135deg,var(--blue),var(--blue2));
  color:#fff;border:none;border-radius:var(--r2);
  padding:12px 24px;font-size:14px;font-weight:700;
  cursor:pointer;box-shadow:var(--sb);transition:var(--tf);
  font-family:'Inter',sans-serif
}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(24,71,181,.4)}
.btn-primary img{width:16px;height:16px}
.btn-secondary{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--w);color:var(--t2);
  border:1.5px solid var(--bd);border-radius:var(--r2);
  padding:10px 20px;font-size:13px;font-weight:500;
  cursor:pointer;transition:var(--tf);font-family:'Inter',sans-serif
}
.btn-secondary:hover{border-color:var(--blue);color:var(--blue);box-shadow:var(--s2)}
.btn-secondary img{width:14px;height:14px;opacity:.6}

/* ── Info/note boxes ── */
.note-box{
  display:flex;align-items:flex-start;gap:8px;
  border-radius:var(--r3);padding:10px 12px
}
.note-blue{background:#eff6ff;border:1px solid #bfdbfe}
.note-amber{background:#fffbeb;border:1px solid #fde68a}
.note-green{background:#f0fdf4;border:1px solid #bbf7d0}
.note-red{background:#fef2f2;border:1px solid #fecaca}
.note-box img{width:14px;height:14px;flex-shrink:0;margin-top:1px}
.note-box span{font-size:11.5px;line-height:1.6}
.note-blue span{color:#1e40af}
.note-amber span{color:#92400e}
.note-green span{color:#065f46}
.note-red span{color:#991b1b}

/* ── Stat cards ── */
.stat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px}
.stat-card{
  background:var(--w);border-radius:var(--r2);border:1px solid var(--bd);
  padding:18px 16px;box-shadow:var(--s1);
  display:flex;align-items:flex-start;gap:12px
}
.stat-icon{
  width:40px;height:40px;border-radius:var(--r3);
  display:flex;align-items:center;justify-content:center;flex-shrink:0
}
.stat-icon img{width:20px;height:20px}
.stat-label{font-size:11px;color:var(--t3);font-weight:500;margin-bottom:4px}
.stat-value{font-size:22px;font-weight:800;line-height:1;margin-bottom:2px}
.stat-sub{font-size:10.5px;font-weight:600}

/* ── Result ring ── */
.result-ring-wrap{position:relative;width:140px;height:140px}
.result-ring-wrap svg{transform:rotate(-90deg)}
.ring-label{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center
}
.ring-pct{font-size:26px;font-weight:800;color:var(--t1)}
.ring-sub{font-size:10px;color:var(--t3)}

/* ── Progress bar ── */
.prog-row{margin-bottom:12px}
.prog-label-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12px;font-weight:500;color:var(--t2)}
.prog-label-row img{width:14px;height:14px}
.prog-track{background:#f1f5f9;border-radius:999px;height:8px;overflow:hidden;flex:1}
.prog-fill{height:100%;border-radius:999px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.prog-pct{font-size:12px;font-weight:700;color:var(--t2);min-width:36px;text-align:right}

/* ── Table ── */
.data-table{width:100%;border-collapse:collapse;font-size:12.5px}
.data-table thead tr{background:#f8fafc}
.data-table th{
  padding:11px 14px;text-align:left;
  font-size:11.5px;font-weight:600;color:var(--t3);
  border-bottom:1px solid var(--bd);white-space:nowrap
}
.data-table th:not(:first-child){text-align:center}
.data-table td{
  padding:11px 14px;border-bottom:1px solid #f1f5f9;
  color:var(--t1);vertical-align:middle
}
.data-table td:not(:first-child){text-align:center}
.data-table tr:last-child td{border:none}
.data-table tr:hover td{background:#f8fafc}
.model-badge-best{
  display:inline-flex;align-items:center;gap:5px;
  font-weight:600;color:var(--blue)
}

/* ── Chart card ── */
.chart-card{
  background:var(--w);border-radius:var(--r);border:1px solid var(--bd);
  padding:20px;box-shadow:var(--s1)
}
.chart-title-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.chart-title{font-size:13.5px;font-weight:700;color:var(--t1)}
.chart-sub{font-size:11px;color:var(--t3);margin-top:2px}
.legend-row{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--t2)}
.legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}

/* ── SHAP waterfall ── */
.shap-row{
  display:flex;align-items:center;gap:10px;
  padding:8px 12px;border-radius:var(--r3);margin-bottom:4px;
  transition:background .15s
}
.shap-row:hover{background:#f8fafc}
.shap-rank{
  width:22px;font-size:10.5px;font-weight:700;
  color:var(--t4);text-align:center;flex-shrink:0
}
.shap-feat{flex:1;font-size:12px;font-weight:500;color:var(--t1);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.shap-inp{
  width:72px;font-size:10.5px;color:var(--t3);
  text-align:right;flex-shrink:0;font-family:monospace
}
.shap-bar-track{
  width:160px;background:#f1f5f9;border-radius:999px;
  height:8px;overflow:hidden;flex-shrink:0
}
.shap-bar-fill{
  height:100%;border-radius:999px;
  transform-origin:left;
  animation:growBar .5s cubic-bezier(.4,0,.2,1) forwards;
  transform:scaleX(0)
}
@keyframes growBar{to{transform:scaleX(1)}}
.shap-val{
  width:64px;font-size:11.5px;font-weight:700;
  text-align:right;flex-shrink:0;font-family:monospace
}
.shap-dir{width:120px;font-size:11px;flex-shrink:0;padding-left:6px;color:var(--t3)}

/* ── SHAP waterfall (patient) ── */
.wf-container{
  background:var(--w);border-radius:var(--r);border:1px solid var(--bd);
  padding:20px;box-shadow:var(--s1)
}
.wf-header{font-size:13px;font-weight:700;color:var(--t1);margin-bottom:6px}
.wf-sub{font-size:11.5px;color:var(--t3);margin-bottom:16px}
.wf-track{position:relative;height:48px;background:#f8fafc;border-radius:var(--r3);overflow:hidden;margin-bottom:10px}
.wf-base{position:absolute;top:0;bottom:0;display:flex;align-items:center;justify-content:center;flex-direction:column}
.wf-block{position:absolute;top:0;bottom:0;display:flex;align-items:center;justify-content:center;flex-direction:column;font-size:10px;font-weight:600;color:#fff}

/* ── About SHAP ── */
.about-shap-card{background:var(--w);border-radius:var(--r);border:1px solid var(--bd);padding:20px;box-shadow:var(--s1)}
.about-item{display:flex;align-items:flex-start;gap:10px;margin-bottom:12px}
.about-item-icon{width:30px;height:30px;border-radius:var(--r3);background:var(--soft);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.about-item-icon img{width:15px;height:15px}
.about-item-title{font-size:12px;font-weight:700;color:var(--t1);margin-bottom:2px}
.about-item-desc{font-size:11px;color:var(--t3);line-height:1.5}

/* ── Summary banner (results) ── */
.result-banner{
  background:var(--w);border-radius:var(--r);border:1px solid var(--bd);
  box-shadow:var(--s1);padding:24px 28px;margin-bottom:20px;
  display:grid;grid-template-columns:200px 160px 1fr 240px;
  gap:24px;align-items:center
}
.rb-divider{width:1px;background:var(--bd);align-self:stretch}

/* ── Confidence gauge ── */
.gauge-wrap{position:relative;width:120px;height:70px;overflow:hidden;margin:0 auto 10px}
.gauge-wrap svg{position:absolute;top:0;left:0;width:120px;height:120px;transform:rotate(180deg)}
.gauge-label{text-align:center}
.gauge-val{font-size:18px;font-weight:800;color:var(--green)}
.gauge-sub{font-size:10px;color:var(--t3)}

/* ── Tab navigation ── */
.tab-nav{display:flex;border-bottom:2px solid var(--bd);margin-bottom:22px;gap:0}
.tab-btn{
  padding:10px 18px;font-size:12.5px;font-weight:500;color:var(--t3);
  cursor:pointer;border:none;background:transparent;
  border-bottom:2px solid transparent;margin-bottom:-2px;
  transition:var(--tf);white-space:nowrap
}
.tab-btn:hover{color:var(--t1)}
.tab-btn.active{color:var(--blue);font-weight:700;border-bottom-color:var(--blue)}

/* ── SHAP top summary strip ── */
.shap-summary-strip{
  display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
  gap:16px;margin-bottom:22px
}
.shap-sum-card{
  background:var(--w);border-radius:var(--r2);border:1px solid var(--bd);
  padding:16px;box-shadow:var(--s1)
}
.shap-sum-label{font-size:11px;color:var(--t3);font-weight:500;margin-bottom:5px}
.shap-sum-val{font-size:20px;font-weight:800;line-height:1.1}
.shap-sum-sub{font-size:10.5px;color:var(--t3);margin-top:3px}

/* ── cols header for SHAP table ── */
.shap-col-header{
  display:flex;align-items:center;gap:10px;
  padding:8px 12px;border-bottom:2px solid var(--bd);
  margin-bottom:6px
}
.shap-col-h{font-size:10.5px;font-weight:600;color:var(--t3)}

/* ── Download prompt ── */
.download-banner{
  background:linear-gradient(135deg,#f8fafc,#eff6ff);
  border:1px solid #dbeafe;border-radius:var(--r2);
  padding:16px 20px;display:flex;align-items:center;
  justify-content:space-between;margin-top:20px
}
.download-banner-text .dt{font-size:14px;font-weight:700;color:var(--t1)}
.download-banner-text .ds{font-size:12px;color:var(--t3);margin-top:2px}
.download-btns{display:flex;gap:10px}

/* ── Security footer ── */
.sec-footer{
  display:flex;align-items:center;justify-content:center;gap:6px;
  padding:14px;font-size:11px;color:var(--t4)
}
.sec-footer img{width:12px;height:12px;opacity:.5}
</style>"""

NAV_JS = """<script>
function nav(k){
  const btns=window.parent.document.querySelectorAll('button[kind="secondary"]');
  const map={home:0,patient:1,results:2,shap:3,perf:4};
  if(btns[map[k]]) btns[map[k]].click();
}
</script>"""

BRAIN_LOGO = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2C10.5 2 9.2 2.8 8.5 4C7.1 4 6 5.1 6 6.5C6 6.8 6.1 7.1 6.2 7.4C5.5 7.9 5 8.8 5 9.8C5 10.5 5.2 11.1 5.6 11.6C5.2 12.1 5 12.8 5 13.5C5 15 6 16.3 7.4 16.8C7.8 18.6 9.3 20 11.1 20.2V22H12.9V20.2C14.7 20 16.2 18.6 16.6 16.8C18 16.3 19 15 19 13.5C19 12.8 18.8 12.1 18.4 11.6C18.8 11.1 19 10.5 19 9.8C19 8.8 18.5 7.9 17.8 7.4C17.9 7.1 18 6.8 18 6.5C18 5.1 16.9 4 15.5 4C14.8 2.8 13.5 2 12 2Z" fill="white" opacity="0.9"/>
<path d="M9 11L10.5 13L12 11L13.5 13L15 11" stroke="white" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.6"/>
</svg>"""

def make_sidebar(active):
    pages = [
        ("home",    "home-24-filled",               "Home"),
        ("patient", "person-24-filled",              "Patient Assessment"),
        ("results", "data-pie-24-filled",            "Prediction Results"),
        ("shap",    "brain-circuit-24-filled",       "Explainability (SHAP)"),
        ("perf",    "arrow-trending-lines-24-filled","Model Performance"),
    ]
    items = ""
    for pid, icon, label in pages:
        cls = "on" if pid == active else ""
        col = "%231847b5" if pid == active else "%2364748b"
        items += f"""<button class="ni {cls}" onclick="nav('{pid}')">
  <span class="ni-icon"><img src="https://api.iconify.design/fluent/{icon}.svg?color={col}"/></span>
  {label}<span class="ni-dot"></span>
</button>"""
    return f"""<aside class="sb">
  <div class="logo-wrap">
    <div class="logo-icon">{BRAIN_LOGO}</div>
    <div class="logo-text">
      <div class="ln">PreSense AI</div>
      <div class="ls">Early Preeclampsia<br>Prediction System</div>
    </div>
  </div>
  <div class="nav-label">Navigation</div>
  <nav>{items}</nav>
  <div class="sb-footer">
    <img src="https://api.iconify.design/fluent/shield-checkmark-24-filled.svg?color=%231847b5" style="width:20px;height:20px;flex-shrink:0;margin-bottom:6px"/>
    <div class="sb-footer-title">Clinically Informed AI<br>for Safer Pregnancies</div>
  </div>
  <div class="ecg"><svg viewBox="0 0 196 36"><polyline
    points="0,18 22,18 32,18 37,4 42,32 47,4 52,18 72,18 92,18 102,18 107,4 112,32 117,4 122,18 196,18"
    stroke="#1847b5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  </svg></div>
</aside>{NAV_JS}"""

def page_shell(active, topbar_title, topbar_sub, topbar_right_html, body_html, h=820):
    sidebar = make_sidebar(active)
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>{FONTS}{BASE_CSS}</head>
<body>
{sidebar}
<div class="main">
  <div class="topbar">
    <div class="topbar-left">
      <div class="topbar-title">{topbar_title}</div>
      <div class="topbar-sub">{topbar_sub}</div>
    </div>
    <div class="topbar-right">{topbar_right_html}</div>
  </div>
  <div class="content">{body_html}</div>
</div>
</body></html>"""
    components.html(html, height=h, scrolling=(h > 860))

# ════════════════════════════════════
# HOME
# ════════════════════════════════════
if cur == "home":
    body = """
<style>
.badge{display:inline-flex;align-items:center;gap:7px;background:var(--w);border:1px solid #d5ddf0;border-radius:999px;padding:6px 16px;font-size:11.5px;font-weight:500;color:var(--t2);margin-bottom:20px;box-shadow:var(--s1)}
.badge img{width:14px;height:14px}
.hero{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:40px;background:transparent;padding:0;border-radius:0;box-shadow:none}
.hero-left{flex:1;max-width:500px}
.hero-title{font-size:48px;font-weight:800;color:var(--t1);line-height:1.05;letter-spacing:-2px;margin-bottom:18px}
.hero-desc{font-size:15px;color:var(--t3);line-height:1.8;max-width:420px;margin-bottom:28px}
.btn-row{display:flex;gap:13px;align-items:center}
.btn-p{display:inline-flex;align-items:center;gap:10px;background:linear-gradient(135deg,#1847b5,#2563eb);color:#fff;border:none;border-radius:13px;padding:13px 26px;font-size:14px;font-weight:700;cursor:pointer;box-shadow:0 4px 18px rgba(24,71,181,.32);transition:all .18s;font-family:'Inter',sans-serif}
.btn-p:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(24,71,181,.42)}
.btn-p .arr{width:28px;height:28px;background:rgba(255,255,255,.2);border-radius:50%;display:flex;align-items:center;justify-content:center}
.btn-p:hover .arr{transform:translateX(2px)}
.btn-p .arr img{width:13px;height:13px}
.hero-right{flex-shrink:0;position:relative;width:420px;height:410px;display:flex;align-items:center;justify-content:center}
.hero-bg-circle{position:absolute;width:390px;height:390px;border-radius:50%;background:linear-gradient(145deg,#dce8fb,#c5d8f8);top:10px;right:5px;box-shadow:0 8px 30px rgba(24,71,181,.14)}
.hero-photo{position:relative;z-index:2;width:370px;height:370px;border-radius:50%;object-fit:cover;object-position:top center;border:5px solid rgba(255,255,255,.9);box-shadow:0 8px 32px rgba(24,71,181,.18);display:block;margin:20px auto 0;transition:all .42s}
.hero-photo:hover{transform:scale(1.015)}
.deco-mol{position:absolute;top:4px;left:10px;width:110px;opacity:.15;z-index:1;pointer-events:none;animation:fl 6s ease-in-out infinite}
.deco-leaf{position:absolute;bottom:8px;right:-8px;width:75px;opacity:.15;z-index:1;pointer-events:none;animation:fl 8s ease-in-out infinite reverse}
@keyframes fl{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}

.feat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.feat-card{background:var(--w);border-radius:var(--r);border:1px solid var(--bd);padding:20px 16px;display:flex;align-items:flex-start;gap:13px;box-shadow:var(--s1);transition:all .28s;position:relative;overflow:hidden}
.feat-card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:18px 18px 0 0;opacity:0;transition:all .18s}
.fc-blue::after{background:linear-gradient(90deg,#1847b5,#60a5fa)}.fc-green::after{background:linear-gradient(90deg,#059669,#34d399)}.fc-purple::after{background:linear-gradient(90deg,#7c3aed,#a78bfa)}.fc-pink::after{background:linear-gradient(90deg,#db2777,#f9a8d4)}
.feat-card:hover{transform:translateY(-4px);box-shadow:0 12px 32px rgba(15,23,42,.12);border-color:var(--bd2)}
.feat-card:hover::after{opacity:1}
.feat-icon{width:50px;height:50px;border-radius:13px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:var(--s1);transition:all .18s}
.feat-card:hover .feat-icon{transform:scale(1.08) rotate(-3deg)}
.fi-blue{background:linear-gradient(135deg,#eff6ff,#dbeafe)}.fi-green{background:linear-gradient(135deg,#ecfdf5,#d1fae5)}.fi-purple{background:linear-gradient(135deg,#f5f3ff,#ede9fe)}.fi-pink{background:linear-gradient(135deg,#fdf2f8,#fce7f3)}
.feat-icon img{width:27px;height:27px;object-fit:contain}
.feat-title{font-size:13px;font-weight:700;color:var(--t1);margin-bottom:5px}
.feat-desc{font-size:11.5px;color:var(--t3);line-height:1.6}
</style>

<div class="hero">
 <div class="hero-left">
  <div class="badge">
    <img src="https://api.iconify.design/fluent/shield-checkmark-24-filled.svg?color=%231847b5"/>
    Evidence-Based &nbsp;•&nbsp; Clinically Validated
  </div>
  <div class="hero-title">AI-Powered Early<br>Preeclampsia<br>Prediction</div>
  <p class="hero-desc">Advanced machine learning using proteomic biomarkers and clinical data to identify high-risk pregnancies early and support better maternal outcomes.</p>
  <div class="btn-row">
   <button class="btn-p" onclick="nav('patient')">
     Start Patient Assessment
     <span class="arr"><img src="https://api.iconify.design/fluent/arrow-right-24-filled.svg?color=white"/></span>
   </button>
  </div>
 </div>
 <div class="hero-right">
  <div class="hero-bg-circle"></div>
  <svg class="deco-mol" viewBox="0 0 120 120">
   <circle cx="60" cy="28" r="9" fill="none" stroke="#1847b5" stroke-width="2.5"/>
   <circle cx="22" cy="78" r="9" fill="none" stroke="#1847b5" stroke-width="2.5"/>
   <circle cx="98" cy="78" r="9" fill="none" stroke="#1847b5" stroke-width="2.5"/>
   <line x1="60" y1="37" x2="30" y2="70" stroke="#1847b5" stroke-width="2"/>
   <line x1="60" y1="37" x2="90" y2="70" stroke="#1847b5" stroke-width="2"/>
   <line x1="31" y1="78" x2="89" y2="78" stroke="#1847b5" stroke-width="2"/>
   <circle cx="60" cy="28" r="4" fill="#1847b5"/>
   <circle cx="22" cy="78" r="4" fill="#1847b5"/>
   <circle cx="98" cy="78" r="4" fill="#1847b5"/>
  </svg>
  <img class="hero-photo"
   src="https://i.pinimg.com/originals/61/cb/e1/61cbe1e96e7e134ba9e19aad05dba9bc.jpg"
   onerror="this.src='https://images.unsplash.com/photo-1491013516836-7db643ee125a?w=700&q=90'"/>
  <svg class="deco-leaf" viewBox="0 0 80 120">
   <path d="M40 115 Q8 82 13 42 Q18 8 40 4 Q62 8 67 42 Q72 82 40 115Z" fill="none" stroke="#1847b5" stroke-width="1.5"/>
   <path d="M40 115 L40 4M40 88 Q22 74 18 56M40 88 Q58 74 62 56" stroke="#1847b5" stroke-width="1" fill="none"/>
  </svg>
 </div>
</div>

<div class="feat-grid">
 <div class="feat-card fc-blue"><div class="feat-icon fi-blue"><img src="https://api.iconify.design/fluent/brain-circuit-24-filled.svg?color=%231847b5"/></div><div><div class="feat-title">AI-Driven Prediction</div><div class="feat-desc">3 ML models trained on <strong>20 proteomic biomarkers</strong> plus clinical variables.</div></div></div>
 <div class="feat-card fc-green"><div class="feat-icon fi-green"><img src="https://api.iconify.design/fluent/target-24-filled.svg?color=%23059669"/></div><div><div class="feat-title">High Sensitivity</div><div class="feat-desc">Optimized threshold ensures <strong style="color:#059669">≥ 90%</strong> sensitivity — catching most at-risk cases.</div></div></div>
 <div class="feat-card fc-purple"><div class="feat-icon fi-purple"><img src="https://api.iconify.design/fluent/data-trending-24-filled.svg?color=%237c3aed"/></div><div><div class="feat-title">Explainable AI</div><div class="feat-desc">SHAP waterfall explanations reveal why each prediction was made.</div></div></div>
 <div class="feat-card fc-pink"><div class="feat-icon fi-pink"><img src="https://api.iconify.design/fluent/heart-pulse-24-filled.svg?color=%23db2777"/></div><div><div class="feat-title">Better Outcomes</div><div class="feat-desc">Early identification enables timely intervention and <strong style="color:#db2777">improved maternal health</strong>.</div></div></div>
</div>
"""
    page_shell("home", "PreSense AI", "Early Preeclampsia Prediction System",
        """<div class="tb-badge"><span class="dot"></span>System Ready</div>""",
        body, h=780)
# ════════════════════════════════════
# PATIENT ASSESSMENT
# ════════════════════════════════════
elif cur == "patient":
    bio_rows_a = ""
    bio_rows_b = ""
    for i, bio in enumerate(SELECTED_BIOMARKERS):
        sl  = bio["sl"]
        nm  = bio["name"][:40] + ("…" if len(bio["name"]) > 40 else "")
        med = bio["median"]
        row = f"""<div class="bio-row">
  <span class="bio-name" title="{bio['name']}">{nm}</span>
  <div class="bio-input-wrap">
    <input id="{sl}" type="number" value="{med}" step="0.1" min="0"/>
    <span class="bio-unit">RFU</span>
  </div>
</div>"""
        if i < 10: bio_rows_a += row
        else:      bio_rows_b += row

    sl_js = "\n".join([f'    + "&{b["sl"]}=" + g("{b["sl"]}")' for b in SELECTED_BIOMARKERS])

    body = f"""
<div class="note-box note-amber" style="margin-bottom:18px">
  <img src="https://api.iconify.design/fluent/warning-24-filled.svg?color=%23d97706"/>
  <span><strong>Important:</strong> Biomarker values must be in <strong>RFU (Relative Fluorescence Units)</strong> as measured by the SomaScan® platform. Defaults shown are population medians from the training cohort.</span>
</div>

<div class="form-grid">
  <!-- Clinical Info -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#eff6ff"><img src="https://api.iconify.design/fluent/calendar-24-filled.svg?color=%231847b5"/></div>
      <div><div class="card-title">Clinical Information</div><div class="card-sub">Patient demographics & obstetric history</div></div>
    </div>
    <div class="card-body">
      <div class="field"><label class="field-label">Maternal Age</label>
        <div class="input-wrap"><input id="age" type="number" value="25" min="15" max="55"/><span class="input-unit">years</span></div></div>
      <div class="field"><label class="field-label">Body Mass Index (BMI)</label>
        <div class="input-wrap"><input id="bmi" type="number" value="27.4" step="0.1" min="15" max="60"/><span class="input-unit">kg/m²</span></div></div>
      <div class="field"><label class="field-label">Gravidity</label>
        <div class="input-wrap"><input id="gravidity" type="number" value="2" min="1" max="10"/><span class="input-unit">total pregnancies</span></div></div>
      <div class="field"><label class="field-label">Parity</label>
        <div class="input-wrap"><input id="parity" type="number" value="1" min="0" max="10"/><span class="input-unit">live births</span></div></div>
      <div class="field"><label class="field-label">Gestational Age at Sampling</label>
        <div class="input-wrap"><input id="ga" type="number" value="11.7" step="0.1" min="4" max="20"/><span class="input-unit">weeks</span></div></div>
      <div class="field"><label class="field-label">Prediction Model</label>
        <div class="select-arrow select-wrap">
          <select id="model_sel">
            <option value="Logistic Regression" selected>Logistic Regression (AUC 0.747)</option>
            <option value="Random Forest">Random Forest (AUC 0.705)</option>
            <option value="XGBoost">XGBoost (AUC 0.739)</option>
          </select>
        </div>
      </div>
      <div class="note-box note-blue">
        <img src="https://api.iconify.design/fluent/info-24-filled.svg?color=%231847b5"/>
        <span>Optimal sampling window: ≤ 20 weeks gestation. Logistic Regression has highest sensitivity (93.8%).</span>
      </div>
    </div>
  </div>

  <!-- Biomarkers A -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#ecfdf5"><img src="https://api.iconify.design/fluent/beaker-24-filled.svg?color=%23059669"/></div>
      <div><div class="card-title">Biomarkers — Part A</div><div class="card-sub">SomaScan RFU values (1–10)</div></div>
    </div>
    <div class="card-body" style="padding:12px 14px">
      {bio_rows_a}
    </div>
  </div>

  <!-- Biomarkers B -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#f5f3ff"><img src="https://api.iconify.design/fluent/beaker-24-filled.svg?color=%237c3aed"/></div>
      <div><div class="card-title">Biomarkers — Part B</div><div class="card-sub">SomaScan RFU values (11–20)</div></div>
    </div>
    <div class="card-body" style="padding:12px 14px">
      {bio_rows_b}
    </div>
  </div>
</div>

<div style="display:flex;justify-content:center">
  <button class="btn-primary" style="width:360px" onclick="submitForm()">
    <img src="https://api.iconify.design/fluent/sparkle-24-filled.svg?color=white"/>
    Generate Prediction
  </button>
</div>

<script>
function g(id){{ return document.getElementById(id).value; }}
function submitForm(){{
  var p = "?action=predict"
    + "&age="       + g("age")
    + "&bmi="       + g("bmi")
    + "&gravidity=" + g("gravidity")
    + "&parity="    + g("parity")
    + "&ga="        + g("ga")
    + "&model="     + encodeURIComponent(g("model_sel"))
{sl_js};
  try{{ window.parent.location.href = window.parent.location.pathname + p; }}
  catch(e){{
    try{{ window.top.location.href = window.top.location.pathname + p; }}
    catch(e2){{ window.location.href = p; }}
  }}
}}
</script>
"""
    page_shell("patient", "Patient Assessment",
        "Enter clinical information and SomaScan proteomic biomarker levels",
        """<div class="tb-badge"><span class="dot"></span>20 Biomarkers</div>""",
        body, h=920)

# ════════════════════════════════════
# RESULTS
# ════════════════════════════════════
elif cur == "results":
    proba     = st.session_state.get("pred_proba")
    risk      = st.session_state.get("pred_risk", "High Risk")
    sel_model = st.session_state.get("pred_model", "Logistic Regression")
    pd_data   = st.session_state.get("patient_data",
        {"Age":25,"BMI":27.4,"Gravidity":2,"Parity":1,"GA":11.7,"n_biomarkers":20,"missing":0})
    stats = MODEL_STATS[sel_model]

    if proba is None:
        body = """<div style="display:flex;align-items:center;justify-content:center;min-height:60vh;flex-direction:column;gap:16px">
  <div style="font-size:48px">🔍</div>
  <div style="font-size:22px;font-weight:800;color:#0f172a">No prediction yet</div>
  <div style="font-size:14px;color:#64748b">Please complete the Patient Assessment form first.</div>
  <button class="btn-primary" onclick="nav('patient')">
    <img src="https://api.iconify.design/fluent/person-24-filled.svg?color=white"/>Go to Patient Assessment
  </button>
</div>"""
        page_shell("results", "Prediction Results", "Run an assessment to see results here",
            "", body, h=600)
    else:
        pct        = int(round(proba * 100))
        no_pct     = 100 - pct
        is_high    = risk == "High Risk"
        risk_color = "#dc2626" if is_high else "#059669"
        risk_icon  = "warning-24-filled" if is_high else "checkmark-circle-24-filled"
        alert_class= "note-red" if is_high else "note-green"
        alert_msg  = ("This patient is at <strong>high risk</strong> of developing early preeclampsia." if is_high
                      else "This patient appears to be at <strong>low risk</strong> for early preeclampsia.")
        recs       = """<div style="margin-top:10px">
  <div class="note-box note-blue"><img src="https://api.iconify.design/fluent/info-24-filled.svg?color=%231847b5"/>
  <span>This is a clinical decision-support tool and does not replace physician judgment.</span></div></div>"""
        if is_high:
            recs = """<div style="margin-top:10px;display:flex;flex-direction:column;gap:7px">
  <div style="display:flex;align-items:center;gap:7px;font-size:12px;color:#334155">
    <img src="https://api.iconify.design/fluent/checkmark-circle-24-filled.svg?color=%23059669" width="14"/> Increase monitoring frequency</div>
  <div style="display:flex;align-items:center;gap:7px;font-size:12px;color:#334155">
    <img src="https://api.iconify.design/fluent/checkmark-circle-24-filled.svg?color=%23059669" width="14"/> Assess blood pressure regularly</div>
  <div style="display:flex;align-items:center;gap:7px;font-size:12px;color:#334155">
    <img src="https://api.iconify.design/fluent/checkmark-circle-24-filled.svg?color=%23059669" width="14"/> Evaluate proteinuria</div>
  <div style="display:flex;align-items:center;gap:7px;font-size:12px;color:#334155">
    <img src="https://api.iconify.design/fluent/checkmark-circle-24-filled.svg?color=%23059669" width="14"/> Consider early intervention</div>
  <div class="note-box note-blue" style="margin-top:6px">
    <img src="https://api.iconify.design/fluent/info-24-filled.svg?color=%231847b5"/>
    <span>Decision-support only — does not replace clinical judgment.</span></div>
</div>"""

        body = f"""
<!-- Result Banner -->
<div class="result-banner">
  <div>
    <div style="font-size:11px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Preeclampsia Risk</div>
    <div style="font-size:28px;font-weight:800;color:{risk_color};margin-bottom:8px">{risk}</div>
    <div style="font-size:12.5px;color:var(--t3);line-height:1.6">{"High risk based on proteomic & clinical data." if is_high else "No elevated risk detected."}</div>
  </div>

  <div style="display:flex;justify-content:center">
    <div class="result-ring-wrap" style="width:140px;height:140px">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r="54" fill="none" stroke="#f1f5f9" stroke-width="12"/>
        <circle cx="70" cy="70" r="54" fill="none" stroke="{risk_color}" stroke-width="12"
          stroke-linecap="round"
          stroke-dasharray="{2*3.14159*54:.1f}"
          stroke-dashoffset="{2*3.14159*54*(1-proba):.1f}"
          transform="rotate(-90 70 70)"/>
      </svg>
      <div class="ring-label" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center">
        <div class="ring-pct">{pct}%</div>
        <div class="ring-sub">Risk</div>
      </div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:14px;justify-content:center">
    <div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;font-size:12px;font-weight:500;color:var(--t2)">
        <span style="display:flex;align-items:center;gap:6px">
          <img src="https://api.iconify.design/fluent/temperature-24-filled.svg?color=%23dc2626" width="14"/>Preeclampsia Probability
        </span><span style="font-weight:700">{pct}%</span>
      </div>
      <div style="background:#f1f5f9;border-radius:999px;height:8px;overflow:hidden">
        <div style="background:linear-gradient(90deg,#dc2626,#f87171);height:100%;border-radius:999px;width:{pct}%"></div>
      </div>
    </div>
    <div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;font-size:12px;font-weight:500;color:var(--t2)">
        <span style="display:flex;align-items:center;gap:6px">
          <img src="https://api.iconify.design/fluent/shield-24-filled.svg?color=%232563eb" width="14"/>No Preeclampsia
        </span><span style="font-weight:700">{no_pct}%</span>
      </div>
      <div style="background:#f1f5f9;border-radius:999px;height:8px;overflow:hidden">
        <div style="background:linear-gradient(90deg,#2563eb,#93c5fd);height:100%;border-radius:999px;width:{no_pct}%"></div>
      </div>
    </div>
  </div>

  <div style="background:#f8fafc;border-radius:var(--r2);border:1px solid var(--bd);padding:16px">
    <div style="font-size:10px;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">MODEL USED</div>
    <div style="font-size:15px;font-weight:800;color:var(--blue);margin-bottom:4px">{sel_model}</div>
    <div style="display:inline-block;background:#d1fae5;color:#065f46;font-size:10.5px;font-weight:600;border-radius:5px;padding:2px 8px;margin-bottom:10px">Clinically Preferred</div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--t2);padding:4px 0;border-bottom:1px solid var(--bd)"><span>AUC</span><span style="font-weight:600">{stats["AUC"]}</span></div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--t2);padding:4px 0;border-bottom:1px solid var(--bd)"><span>Sensitivity</span><span style="font-weight:600">{stats["Sens"]}</span></div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--t2);padding:4px 0;border-bottom:1px solid var(--bd)"><span>Specificity</span><span style="font-weight:600">{stats["Spec"]}</span></div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--t2);padding:4px 0"><span>F1 Score</span><span style="font-weight:600">{stats["F1"]}</span></div>
  </div>
</div>

<!-- 3 bottom cards -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:20px">

  <!-- Risk Interpretation -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#eff6ff"><img src="https://api.iconify.design/fluent/chat-warning-24-filled.svg?color=%231847b5"/></div>
      <div><div class="card-title">Risk Interpretation</div></div>
    </div>
    <div class="card-body">
      <div class="note-box {alert_class}">
        <img src="https://api.iconify.design/fluent/{risk_icon}.svg?color={risk_color}" width="14"/>
        <span>{alert_msg}</span>
      </div>
      {recs}
    </div>
  </div>

  <!-- Probability Overview -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#ecfdf5"><img src="https://api.iconify.design/fluent/arrow-trending-24-filled.svg?color=%23059669"/></div>
      <div><div class="card-title">Probability Overview</div></div>
    </div>
    <div class="card-body" style="display:flex;flex-direction:column;align-items:center">
      <div style="position:relative;width:130px;height:130px;margin-bottom:14px">
        <svg width="130" height="130" viewBox="0 0 130 130">
          <circle cx="65" cy="65" r="50" fill="none" stroke="#2563eb" stroke-width="22"/>
          <circle cx="65" cy="65" r="50" fill="none" stroke="#dc2626" stroke-width="22"
            stroke-dasharray="{2*3.14159*50:.1f}"
            stroke-dashoffset="{2*3.14159*50*(1-proba):.1f}"
            transform="rotate(-90 65 65)"/>
        </svg>
      </div>
      <div style="width:100%;display:flex;flex-direction:column;gap:6px">
        <div style="display:flex;align-items:center;justify-content:space-between;font-size:12px;color:var(--t2)">
          <span style="display:flex;align-items:center;gap:6px"><span style="width:8px;height:8px;border-radius:50%;background:#dc2626;display:inline-block"></span>Preeclampsia</span>
          <strong>{pct}%</strong>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;font-size:12px;color:var(--t2)">
          <span style="display:flex;align-items:center;gap:6px"><span style="width:8px;height:8px;border-radius:50%;background:#2563eb;display:inline-block"></span>No Preeclampsia</span>
          <strong>{no_pct}%</strong>
        </div>
        <div style="font-size:11px;color:var(--t3);margin-top:6px;line-height:1.5">The model estimates the probability of preeclampsia based on clinical factors and biomarker levels.</div>
      </div>
    </div>
  </div>

  <!-- Key Input Summary -->
  <div class="card">
    <div class="card-header">
      <div class="card-icon" style="background:#f5f3ff"><img src="https://api.iconify.design/fluent/document-bullet-list-24-filled.svg?color=%237c3aed"/></div>
      <div><div class="card-title">Key Input Summary</div></div>
    </div>
    <div class="card-body">
      {"".join(f'<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;font-size:12.5px"><span style="display:flex;align-items:center;gap:6px;color:var(--t3)"><img src="https://api.iconify.design/fluent/{ic}.svg?color=%2394a3b8" width="13"/>{lbl}</span><strong style="color:var(--t1)">{val}</strong></div>' for lbl,val,ic in [
          ("Maternal Age", f'{pd_data.get("Age","—")} years', "calendar-24-regular"),
          ("BMI", f'{pd_data.get("BMI","—")} kg/m²', "person-24-regular"),
          ("Gravidity", pd_data.get("Gravidity","—"), "people-24-regular"),
          ("Gestational Age", f'{pd_data.get("GA","—")} weeks', "calendar-clock-24-regular"),
          ("Biomarkers", f'{pd_data.get("n_biomarkers",20)} / 20', "beaker-24-regular"),
          ("Missing → Median", pd_data.get("missing",0), "warning-24-regular"),
      ])}
    </div>
  </div>
</div>

<!-- SHAP prompt -->
<div style="background:linear-gradient(135deg,#f8fafc,#eff6ff);border:1px solid #dbeafe;border-radius:var(--r2);padding:16px 22px;display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="width:40px;height:40px;border-radius:var(--r3);background:var(--soft);display:flex;align-items:center;justify-content:center">
      <img src="https://api.iconify.design/fluent/sparkle-24-filled.svg?color=%231847b5" width="20"/>
    </div>
    <div>
      <div style="font-size:14px;font-weight:700;color:var(--t1)">Want to understand how the model reached this prediction?</div>
      <div style="font-size:12px;color:var(--t3)">Explore the key factors and their impact using SHAP explainability.</div>
    </div>
  </div>
  <button class="btn-secondary" onclick="nav('shap')" style="border-color:#bfdbfe;color:var(--blue);white-space:nowrap">
    View SHAP Explanation →
  </button>
</div>

<!-- Actions -->
<div style="display:flex;align-items:center;justify-content:center;gap:14px">
  <button class="btn-secondary" onclick="nav('patient')">
    <img src="https://api.iconify.design/fluent/arrow-counterclockwise-24-filled.svg?color=%2364748b"/>New Assessment
  </button>
</div>
<div class="sec-footer">
  <img src="https://api.iconify.design/fluent/lock-closed-24-filled.svg?color=%2394a3b8"/>
  Your data is secure and confidential
</div>
"""
        page_shell("results", "Prediction Results",
            "AI-generated risk prediction based on clinical and proteomic biomarker inputs",
            f"""<div class="tb-badge"><span class="dot"></span>Evidence-Based &nbsp;•&nbsp; Clinically Validated</div>
            <div class="tb-help"><img src="https://api.iconify.design/fluent/question-circle-24-regular.svg?color=%2364748b"/></div>""",
            body, h=940)

# ════════════════════════════════════
# SHAP
# ════════════════════════════════════
elif cur == "shap":
    shap_vals  = st.session_state.get("shap_values")
    shap_feats = st.session_state.get("shap_features")
    shap_inps  = st.session_state.get("shap_inputs")
    shap_base  = st.session_state.get("shap_base")
    sel_model  = st.session_state.get("pred_model", "Logistic Regression")
    proba      = st.session_state.get("pred_proba")
    risk       = st.session_state.get("pred_risk", "—")

    if shap_vals is None or proba is None:
        body = """<div style="display:flex;align-items:center;justify-content:center;min-height:60vh;flex-direction:column;gap:16px">
  <div style="font-size:48px">🧬</div>
  <div style="font-size:22px;font-weight:800;color:#0f172a">No prediction yet</div>
  <div style="font-size:14px;color:#64748b">Run a patient assessment first to see the SHAP explanation.</div>
  <button class="btn-primary" onclick="nav('patient')">
    <img src="https://api.iconify.design/fluent/person-24-filled.svg?color=white"/>Go to Patient Assessment
  </button>
</div>"""
        page_shell("shap", "Explainability (SHAP)", "", "", body, h=580)
    else:
        pct_proba  = int(round(proba * 100))
        is_high    = risk == "High Risk"
        risk_color = "#dc2626" if is_high else "#059669"
        max_abs    = max(abs(v) for v in shap_vals) or 1.0

        # count directions
        n_pos = sum(1 for v in shap_vals if v > 0)
        n_neg = sum(1 for v in shap_vals if v < 0)

        # SHAP feature bars
        bars_html = ""
        for i, (feat, val, inp) in enumerate(zip(shap_feats, shap_vals, shap_inps)):
            is_pos  = val >= 0
            color   = "#dc2626" if is_pos else "#059669"
            bg      = "#fef2f2" if is_pos else "#f0fdf4"
            bar_w   = min(abs(val) / max_abs * 100, 100)
            sign    = f"+{val:.3f}" if is_pos else f"{val:.3f}"
            arrow   = "↑ increases risk" if is_pos else "↓ decreases risk"
            fname   = feat[:38] + ("…" if len(feat) > 38 else "")
            bars_html += f"""<div class="shap-row" style="background:{bg}">
  <span class="shap-rank">{i+1}</span>
  <span class="shap-feat" title="{feat}">{fname}</span>
  <span class="shap-inp">{inp:.2f}</span>
  <div class="shap-bar-track">
    <div class="shap-bar-fill" style="width:{bar_w:.1f}%;background:{color};animation-delay:{i*0.04}s"></div>
  </div>
  <span class="shap-val" style="color:{color}">{sign}</span>
  <span class="shap-dir" style="color:{color}">{arrow}</span>
</div>"""

        # Waterfall strip (simplified)
        wf_items = list(zip(shap_feats[:6], shap_vals[:6]))
        total_range = abs(shap_base or 0) + sum(abs(v) for _, v in wf_items) + 0.2
        track_w = 560
        base_px = (abs(shap_base or 0) / total_range) * track_w if shap_base else 40
        wf_blocks = ""
        cursor = base_px
        for feat, val in wf_items:
            bw = max(2, abs(val) / total_range * track_w)
            color = "#dc2626" if val >= 0 else "#059669"
            name  = feat.split("(")[0][:16].strip()
            sign  = f"+{val:.2f}" if val >= 0 else f"{val:.2f}"
            wf_blocks += f"""<div style="position:absolute;left:{cursor:.1f}px;width:{bw:.1f}px;top:0;bottom:0;
              background:{color};display:flex;align-items:center;justify-content:center;
              flex-direction:column;overflow:hidden">
              <span style="font-size:8.5px;font-weight:700;color:#fff;text-align:center;line-height:1.1;padding:0 2px;white-space:nowrap">{sign}</span>
              <span style="font-size:7px;color:rgba(255,255,255,.8);white-space:nowrap;padding:0 2px;overflow:hidden;max-width:{bw:.0f}px">{name}</span>
            </div>"""
            cursor += bw

        body = f"""
<!-- Top summary strip -->
<div class="shap-summary-strip">
  <div class="shap-sum-card">
    <div class="shap-sum-label">Prediction Outcome</div>
    <div class="shap-sum-val" style="color:{risk_color}">{risk}</div>
    <div class="shap-sum-sub">Risk Probability: {pct_proba}%</div>
  </div>
  <div class="shap-sum-card">
    <div class="shap-sum-label">Model Confidence</div>
    <div class="shap-sum-val" style="color:{'#dc2626' if pct_proba>70 else '#059669'}">{pct_proba}%</div>
    <div class="shap-sum-sub">Base value: {shap_base:.3f}</div>
  </div>
  <div class="shap-sum-card">
    <div class="shap-sum-label">Top Contributing Direction</div>
    <div style="margin-top:4px">
      <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#dc2626;font-weight:600;margin-bottom:4px">
        <img src="https://api.iconify.design/fluent/arrow-up-24-filled.svg?color=%23dc2626" width="13"/> Increased Risk — {n_pos} factors
      </div>
      <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#059669;font-weight:600">
        <img src="https://api.iconify.design/fluent/arrow-down-24-filled.svg?color=%23059669" width="13"/> Decreased Risk — {n_neg} factors
      </div>
    </div>
  </div>
  <div class="shap-sum-card">
    <div class="shap-sum-label">Top Driver</div>
    <div class="shap-sum-val" style="font-size:14px;padding-top:4px">{shap_feats[0][:24]}{"…" if len(shap_feats[0])>24 else ""}</div>
    <div class="shap-sum-sub" style="color:{'#dc2626' if shap_vals[0]>=0 else '#059669'}">SHAP = {shap_vals[0]:+.3f}</div>
  </div>
</div>

<!-- Main 2-col layout -->
<div style="display:grid;grid-template-columns:1fr 340px;gap:20px;margin-bottom:20px">

  <!-- Left: Feature table -->
  <div class="chart-card">
    <div class="chart-title-row">
      <div>
        <div class="chart-title">Top Contributing Features</div>
        <div class="chart-sub" style="margin-top:3px">Sorted by |SHAP| · {sel_model}</div>
      </div>
      <div style="font-size:11px;color:var(--t3)">SHAP Value (Impact on Risk)</div>
    </div>
    <div class="shap-col-header">
      <span class="shap-col-h" style="width:22px;text-align:center">#</span>
      <span class="shap-col-h" style="flex:1">Feature</span>
      <span class="shap-col-h" style="width:72px;text-align:right">Scaled val</span>
      <span class="shap-col-h" style="width:160px">SHAP bar</span>
      <span class="shap-col-h" style="width:64px;text-align:right">SHAP</span>
      <span class="shap-col-h" style="width:120px">Direction</span>
    </div>
    {bars_html}
    <div class="legend-row">
      <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>Increases risk</div>
      <div class="legend-item"><span class="legend-dot" style="background:#059669"></span>Decreases risk</div>
    </div>
  </div>

  <!-- Right: About SHAP -->
  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="about-shap-card">
      <div class="chart-title" style="margin-bottom:14px">About SHAP</div>
      <div style="font-size:12px;color:var(--t3);line-height:1.6;margin-bottom:14px">
        SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model.
      </div>
      <div class="about-item">
        <div class="about-item-icon"><img src="https://api.iconify.design/fluent/checkmark-circle-24-filled.svg?color=%231847b5"/></div>
        <div><div class="about-item-title">Consistent</div><div class="about-item-desc">Provides consistent and locally accurate explanations.</div></div>
      </div>
      <div class="about-item">
        <div class="about-item-icon"><img src="https://api.iconify.design/fluent/globe-24-filled.svg?color=%231847b5"/></div>
        <div><div class="about-item-title">Global + Local</div><div class="about-item-desc">Explains both overall model behavior and individual predictions.</div></div>
      </div>
      <div class="about-item" style="margin-bottom:0">
        <div class="about-item-icon"><img src="https://api.iconify.design/fluent/brain-circuit-24-filled.svg?color=%231847b5"/></div>
        <div><div class="about-item-title">Model Agnostic</div><div class="about-item-desc">Works with any machine learning algorithm.</div></div>
      </div>
    </div>
    <div class="note-box note-amber" style="border-radius:var(--r2)">
      <img src="https://api.iconify.design/fluent/warning-24-filled.svg?color=%23d97706"/>
      <span>SHAP values explain the model's internal decision — not direct clinical causation. Always combine with clinical judgment.</span>
    </div>
  </div>
</div>

<!-- Feature Impact waterfall -->
<div class="wf-container">
  <div class="wf-header">Feature Impact for This Patient</div>
  <div class="wf-sub">How each feature shifted the prediction from the base value to the final risk score</div>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <div style="font-size:10px;color:var(--t3);white-space:nowrap">Base Value<br>(avg prediction)</div>
    <div style="position:relative;flex:1;height:44px;background:#f1f5f9;border-radius:var(--r3);overflow:hidden">
      <div style="position:absolute;left:0;top:0;bottom:0;width:{base_px:.1f}px;
        background:#94a3b8;display:flex;align-items:center;justify-content:center">
        <span style="font-size:9px;font-weight:700;color:#fff">{shap_base:.2f}</span>
      </div>
      {wf_blocks}
    </div>
    <div style="font-size:10px;color:var(--t3);white-space:nowrap;text-align:right">
      Final Prediction<br>(Preeclampsia Risk)<br>
      <strong style="font-size:18px;color:{risk_color}">{proba:.2f}</strong>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10.5px;color:var(--t3);margin-top:6px">
    <span>← Lower Risk</span><span>Higher Risk →</span>
  </div>
</div>
"""
        page_shell("shap", "Explainability (SHAP)",
            "Understand the key biomarkers and clinical factors that contributed to this prediction",
            f"""<div class="tb-badge" style="gap:8px">
              <span style="width:8px;height:8px;border-radius:50%;background:var(--blue);display:inline-block"></span>
              Model Used: {sel_model}
            </div>
            <div class="tb-help"><img src="https://api.iconify.design/fluent/question-circle-24-regular.svg?color=%2364748b"/></div>""",
            body, h=1100)

# ════════════════════════════════════
# MODEL PERFORMANCE
# ════════════════════════════════════
elif cur == "perf":
    perf_data = model_info.get("perf_data", {})
    MC = {
        "Logistic Regression": {"color":"#1847b5","color2":"#3b82f6","short":"LR"},
        "Random Forest":       {"color":"#059669","color2":"#34d399","short":"RF"},
        "XGBoost":             {"color":"#d97706","color2":"#fbbf24","short":"XGB"},
    }

    if not perf_data:
        body = """<div style="display:flex;align-items:center;justify-content:center;min-height:60vh;flex-direction:column;gap:16px">
  <div style="font-size:48px">📊</div>
  <div style="font-size:22px;font-weight:800;color:#0f172a">Performance data not found</div>
  <div style="font-size:14px;color:#64748b;text-align:center;max-width:400px">Re-export perf_data from the notebook and re-upload model_info.json.</div>
</div>"""
        page_shell("perf", "Model Performance", "", "", body, h=580)
    else:
        best_model = max(perf_data.keys(), key=lambda n: perf_data[n].get("AUC", 0))
        best = perf_data[best_model]

        # stat cards
        stat_cards = f"""<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-icon" style="background:#eff6ff"><img src="https://api.iconify.design/fluent/arrow-trending-lines-24-filled.svg?color=%231847b5"/></div>
    <div><div class="stat-label">Best AUC (Test)</div>
      <div class="stat-value" style="color:#1847b5">{best.get('AUC',0):.3f}</div>
      <div class="stat-sub" style="color:#1847b5">{best_model}</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#ecfdf5"><img src="https://api.iconify.design/fluent/target-24-filled.svg?color=%23059669"/></div>
    <div><div class="stat-label">Sensitivity</div>
      <div class="stat-value" style="color:#059669">{best.get('Sens',0)*100:.1f}%</div>
      <div class="stat-sub" style="color:#059669">Excellent</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#f5f3ff"><img src="https://api.iconify.design/fluent/shield-24-filled.svg?color=%237c3aed"/></div>
    <div><div class="stat-label">Specificity</div>
      <div class="stat-value" style="color:#7c3aed">{best.get('Spec',0)*100:.1f}%</div>
      <div class="stat-sub" style="color:#7c3aed">Good</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#fffbeb"><img src="https://api.iconify.design/fluent/star-24-filled.svg?color=%23d97706"/></div>
    <div><div class="stat-label">F1 Score</div>
      <div class="stat-value" style="color:#d97706">{best.get('F1',0):.3f}</div>
      <div class="stat-sub" style="color:#d97706">Good</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#fef2f2"><img src="https://api.iconify.design/fluent/checkmark-circle-24-filled.svg?color=%23dc2626"/></div>
    <div><div class="stat-label">Accuracy</div>
      <div class="stat-value" style="color:#dc2626">{best.get('Acc',0)*100:.1f}%</div>
      <div class="stat-sub" style="color:#dc2626">Good</div></div>
  </div>
</div>"""

        # ── SVG helpers ──
        W, H, PAD = 420, 280, 38

        def to_pts(xs, ys):
            out = []
            for x, y in zip(xs, ys):
                px = PAD + x * (W - 2*PAD)
                py = (H - PAD) - y * (H - 2*PAD)
                out.append(f"{px:.1f},{py:.1f}")
            return " ".join(out)

        # Axis grid lines
        def axis_grid():
            lines = ""
            for v in [0.25, 0.5, 0.75]:
                gx = PAD + v*(W-2*PAD)
                gy = (H-PAD) - v*(H-2*PAD)
                lines += f'<line x1="{gx:.0f}" y1="{PAD}" x2="{gx:.0f}" y2="{H-PAD}" stroke="#f1f5f9" stroke-width="1"/>'
                lines += f'<line x1="{PAD}" y1="{gy:.0f}" x2="{W-PAD}" y2="{gy:.0f}" stroke="#f1f5f9" stroke-width="1"/>'
                lines += f'<text x="{gx:.0f}" y="{H-PAD+12}" font-size="8" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">{v}</text>'
                lines += f'<text x="{PAD-6}" y="{gy:.0f}" font-size="8" fill="#94a3b8" text-anchor="end" dominant-baseline="middle" font-family="Inter,sans-serif">{v}</text>'
            return lines

        # ROC
        diag_pts = to_pts([0,1],[0,1])
        roc_curves = ""
        roc_legend = ""
        for nm, mc in MC.items():
            if nm not in perf_data: continue
            d   = perf_data[nm]
            pts = to_pts(d.get("roc_fpr",[0,1]), d.get("roc_tpr",[0,1]))
            c   = mc["color"]
            auc = d.get("AUC",0)
            roc_curves += f'<polyline points="{PAD},{H-PAD} {pts} {W-PAD},{H-PAD}" fill="{c}" fill-opacity="0.07" stroke="none"/>'
            roc_curves += f'<polyline points="{pts}" stroke="{c}" stroke-width="2" fill="none" stroke-linejoin="round"/>'
            roc_legend  += f'<div class="legend-item"><span class="legend-dot" style="background:{c}"></span>{nm} (AUC={auc:.3f})</div>'

        roc_svg = f"""<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">
  <rect x="{PAD}" y="{PAD}" width="{W-2*PAD}" height="{H-2*PAD}" fill="#fafafa" rx="2"/>
  {axis_grid()}
  <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#e2e8f0" stroke-width="1"/>
  <line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#e2e8f0" stroke-width="1"/>
  <polyline points="{diag_pts}" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="5,4" fill="none"/>
  {roc_curves}
  <text x="{W//2}" y="{H-4}" font-size="9" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">False Positive Rate (1 - Specificity)</text>
  <text x="12" y="{H//2}" font-size="9" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif" transform="rotate(-90,12,{H//2})">True Positive Rate (Sensitivity)</text>
</svg>"""

        # PR
        pr_curves = ""
        pr_legend = ""
        for nm, mc in MC.items():
            if nm not in perf_data: continue
            d   = perf_data[nm]
            pts = to_pts(d.get("pr_rec",[0,1]), d.get("pr_prec",[1,0]))
            c   = mc["color"]
            ap  = d.get("AUPRC", 0)
            pr_curves += f'<polyline points="{pts}" stroke="{c}" stroke-width="2" fill="none" stroke-linejoin="round"/>'
            pr_legend  += f'<div class="legend-item"><span class="legend-dot" style="background:{c}"></span>{mc["short"]} (AP={ap:.3f})</div>'

        pr_svg = f"""<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">
  <rect x="{PAD}" y="{PAD}" width="{W-2*PAD}" height="{H-2*PAD}" fill="#fafafa" rx="2"/>
  {axis_grid()}
  <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#e2e8f0" stroke-width="1"/>
  <line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#e2e8f0" stroke-width="1"/>
  {pr_curves}
  <text x="{W//2}" y="{H-4}" font-size="9" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">Recall (Sensitivity)</text>
  <text x="12" y="{H//2}" font-size="9" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif" transform="rotate(-90,12,{H//2})">Precision</text>
</svg>"""

        # Nested CV bar chart
        cv_W, cv_H, cv_PAD = 700, 220, 36
        bar_section_w = (cv_W - 2*cv_PAD) / 3
        cv_bars = ""
        cv_legend = ""
        for mi, (nm, mc) in enumerate(MC.items()):
            if nm not in perf_data: continue
            d     = perf_data[nm]
            folds = d.get("cv_folds_auc", [])
            mean  = d.get("AUC_CV", 0)
            std   = d.get("AUC_CV_std", 0)
            c     = mc["color"]
            sh    = mc["short"]
            n     = len(folds)
            sec_x = cv_PAD + mi * bar_section_w
            bw    = (bar_section_w - 20) / max(n, 1)
            for fi, av in enumerate(folds):
                bh = max(2, (av - 0.4) / 0.6 * (cv_H - cv_PAD - 20))
                bx = sec_x + 10 + fi * bw
                by = cv_H - cv_PAD - bh
                cv_bars += f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-4:.1f}" height="{bh:.1f}" rx="3" fill="{c}" opacity=".8"/>'
                cv_bars += f'<text x="{bx+(bw-4)/2:.1f}" y="{by-3:.1f}" font-size="7.5" fill="{c}" text-anchor="middle" font-family="Inter,sans-serif">{av:.3f}</text>'
                cv_bars += f'<text x="{bx+(bw-4)/2:.1f}" y="{cv_H-cv_PAD+12}" font-size="7.5" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">F{fi+1}</text>'
            # mean dashed line
            my = cv_H - cv_PAD - max(2, (mean-0.4)/0.6*(cv_H-cv_PAD-20))
            cv_bars += f'<line x1="{sec_x+5}" y1="{my:.1f}" x2="{sec_x+bar_section_w-5}" y2="{my:.1f}" stroke="{c}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            cv_bars += f'<text x="{sec_x+bar_section_w/2:.1f}" y="{my-4:.1f}" font-size="8" fill="{c}" text-anchor="middle" font-family="Inter,sans-serif">{sh}: μ={mean:.3f}±{std:.3f}</text>'
            cv_legend += f'<div class="legend-item"><span class="legend-dot" style="background:{c}"></span>{nm}</div>'

        # y-axis labels
        cv_yaxis = ""
        for yv in [0.4, 0.6, 0.8, 1.0]:
            gy = cv_H - cv_PAD - (yv-0.4)/0.6*(cv_H-cv_PAD-20)
            cv_yaxis += f'<line x1="{cv_PAD}" y1="{gy:.0f}" x2="{cv_W-10}" y2="{gy:.0f}" stroke="#f1f5f9" stroke-width="1"/>'
            cv_yaxis += f'<text x="{cv_PAD-4}" y="{gy:.0f}" font-size="7.5" fill="#94a3b8" text-anchor="end" dominant-baseline="middle" font-family="Inter,sans-serif">{yv}</text>'

        cv_svg = f"""<svg viewBox="0 0 {cv_W} {cv_H}" style="width:100%;height:auto">
  <line x1="{cv_PAD}" y1="8" x2="{cv_PAD}" y2="{cv_H-cv_PAD}" stroke="#e2e8f0" stroke-width="1"/>
  <line x1="{cv_PAD}" y1="{cv_H-cv_PAD}" x2="{cv_W-10}" y2="{cv_H-cv_PAD}" stroke="#e2e8f0" stroke-width="1"/>
  {cv_yaxis}
  {cv_bars}
  <text x="{cv_W//2}" y="{cv_H-2}" font-size="9" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">Fold</text>
  <text x="10" y="{cv_H//2}" font-size="9" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif" transform="rotate(-90,10,{cv_H//2})">AUC</text>
</svg>"""

        # Model comparison table
        table_rows = ""
        for nm, mc in MC.items():
            if nm not in perf_data: continue
            d = perf_data[nm]
            c = mc["color"]
            star = "⭐ " if nm == best_model else ""
            cells = f'<td><span style="color:{c};font-weight:600">{star}{nm}</span></td>'
            cells += f'<td style="font-weight:{"700" if nm==best_model else "400"};color:{""+c if nm==best_model else "inherit"}">{d.get("AUC",0):.3f}</td>'
            cells += f'<td>{d.get("Sens",0)*100:.1f}%</td>'
            cells += f'<td>{d.get("Spec",0)*100:.1f}%</td>'
            cells += f'<td>{d.get("F1",0):.3f}</td>'
            cells += f'<td>{d.get("Acc",0)*100:.1f}%</td>'
            table_rows += f"<tr>{cells}</tr>"

        # CV summary stats
        lr_cv = perf_data.get("Logistic Regression", {})
        cv_mean = lr_cv.get("AUC_CV", 0)
        cv_std  = lr_cv.get("AUC_CV_std", 0)
        cv_sens = lr_cv.get("Sens", 0)
        cv_spec = lr_cv.get("Spec", 0)
        cv_f1   = lr_cv.get("F1", 0)

        body = f"""
{stat_cards}

<!-- Tabbed section header -->
<div class="tab-nav" id="perf-tabs">
  <button class="tab-btn active" onclick="showTab('roc',this)">ROC Curve</button>
  <button class="tab-btn" onclick="showTab('pr',this)">Precision-Recall Curve</button>
  <button class="tab-btn" onclick="showTab('cv',this)">Nested Cross-Validation</button>
  <button class="tab-btn" onclick="showTab('cmp',this)">Model Comparison</button>
</div>

<!-- ROC tab -->
<div id="tab-roc" class="tab-pane" style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:22px">
  <div class="chart-card">
    <div class="chart-title-row">
      <div><div class="chart-title">ROC Curve (Test Set)</div></div>
    </div>
    {roc_svg}
    <div class="legend-row">
      {roc_legend}
      <div class="legend-item"><span style="display:inline-block;width:18px;border-top:1.5px dashed #cbd5e1;margin-right:4px"></span>Baseline (AUC=0.500)</div>
    </div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Model Comparison (Test Set)</div>
    <table class="data-table" style="margin-top:10px">
      <thead><tr><th>Model</th><th>AUC</th><th>Sensitivity</th><th>Specificity</th><th>F1 Score</th><th>Accuracy</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div class="note-box note-blue" style="margin-top:12px">
      <img src="https://api.iconify.design/fluent/info-24-filled.svg?color=%231847b5"/>
      <span>⭐ {best_model} achieved the best AUC and sensitivity on the test set.</span>
    </div>
  </div>
</div>

<!-- PR tab -->
<div id="tab-pr" class="tab-pane" style="display:none;margin-bottom:22px">
  <div class="chart-card" style="max-width:520px">
    <div class="chart-title">Precision-Recall Curve (Test Set)</div>
    {pr_svg}
    <div class="legend-row">{pr_legend}</div>
  </div>
</div>

<!-- CV tab -->
<div id="tab-cv" class="tab-pane" style="display:none;margin-bottom:22px">
  <div class="chart-card">
    <div class="chart-title-row">
      <div>
        <div class="chart-title">Nested Cross-Validation (5×5)</div>
        <div class="chart-sub">AUC across folds — all 3 models</div>
      </div>
    </div>
    <!-- CV summary mini stats -->
    <div style="display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap">
      <div><div style="font-size:10px;color:var(--t3);font-weight:500">Mean AUC (± SD)</div>
        <div style="font-size:18px;font-weight:800;color:#1847b5">{cv_mean:.3f} ± {cv_std:.3f}</div></div>
      <div><div style="font-size:10px;color:var(--t3);font-weight:500">Mean Sensitivity</div>
        <div style="font-size:18px;font-weight:800;color:#059669">{cv_sens*100:.1f}%</div></div>
      <div><div style="font-size:10px;color:var(--t3);font-weight:500">Mean Specificity</div>
        <div style="font-size:18px;font-weight:800;color:#7c3aed">{cv_spec*100:.1f}%</div></div>
      <div><div style="font-size:10px;color:var(--t3);font-weight:500">Mean F1 Score</div>
        <div style="font-size:18px;font-weight:800;color:#d97706">{cv_f1:.3f}</div></div>
    </div>
    {cv_svg}
    <div class="legend-row">{cv_legend}</div>
  </div>
</div>

<!-- CMP tab -->
<div id="tab-cmp" class="tab-pane" style="display:none;margin-bottom:22px">
  <div class="chart-card">
    <div class="chart-title" style="margin-bottom:14px">Full Model Comparison (Test Set)</div>
    <table class="data-table">
      <thead><tr><th>Model</th><th>AUC</th><th>Sensitivity</th><th>Specificity</th><th>F1 Score</th><th>Accuracy</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div class="note-box note-blue" style="margin-top:14px">
      <img src="https://api.iconify.design/fluent/info-24-filled.svg?color=%231847b5"/>
      <span>Performance metrics are calculated on the independent test set. Nested cross-validation was used for robust model evaluation.</span>
    </div>
  </div>
</div>

<script>
function showTab(id, btn) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).style.display = id === 'roc' ? 'grid' : 'block';
  btn.classList.add('active');
}}
</script>
"""
        page_shell("perf", "Model Performance",
            "Evaluate and compare the performance of machine learning models for early preeclampsia prediction",
            f"""<div class="tb-badge" style="gap:8px">
              <span style="color:#1847b5">⭐</span> Best Model: <strong style="color:#1847b5">{best_model}</strong>
            </div>
            <div class="tb-help"><img src="https://api.iconify.design/fluent/question-circle-24-regular.svg?color=%2364748b"/></div>""",
            body, h=1020)
