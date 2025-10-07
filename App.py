import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from pathlib import Path
from xgboost import XGBRegressor  # để tương thích bundle nếu dùng XGB
import altair as alt

# ================== LOAD MODEL ==================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "thebestmodel.pkl")

@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)

bundle  = load_bundle(MODEL_PATH)
model   = bundle["model"]
scalerX = bundle.get("scaler_X", None)
scalerY = bundle.get("scaler_y", None)

# ================== PAGE & THEME ==================
st.set_page_config(page_title="N Predictor", page_icon="🧮", layout="centered")

st.markdown("""
<style>
.stApp { background:#ffffff !important; color:#000000 !important; }
header[data-testid="stHeader"] { display:none; }
.block-container { padding-top: 1rem; }
.stNumberInput input { background:#fff !important; color:#000 !important; border:1px solid #00000040 !important; border-radius:6px !important; }
.stButton > button, .stDownloadButton > button { background:#fff !important; color:#000 !important; border:1px solid #000 !important; border-radius:6px !important; }
.hero-img { max-width: 100%; height: auto; object-fit: contain; display: block; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

st.title("Problem Definition")

# ================== SINGLE TOP BANNER ==================
st.sidebar.header("Banner settings")
hero_height_px = st.sidebar.slider("Banner max height (px)", min_value=120, max_value=400, value=400, step=10)
hero_name = st.sidebar.text_input("Image filename", value="Problem_definition.svg")

root_dir = Path(__file__).parent
hero_path = (root_dir / hero_name).resolve()

def _data_url(path: Path):
    if not path.exists():
        return None, None
    ext = path.suffix.lower()
    mime = {".svg":"image/svg+xml",".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp"}.get(ext)
    if not mime: return None, None
    b64 = base64.b64encode(path.read_bytes()).decode()
    return mime, b64

def _img_html(path: Path, css_class: str, max_h: int | None = None):
    mime, b64 = _data_url(path)
    if not mime: return ""
    style = f"max-height:{max_h}px;" if max_h else ""
    return f'<img class="{css_class}" src="data:{mime};base64,{b64}" alt="{path.name}" style="{style}">'

def render_hero(path: Path, max_height_px: int):
    html = _img_html(path, "hero-img", max_h=max_height_px)
    if not html:
        st.warning("Banner image missing or unsupported. Check filename in the sidebar.")
        return
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:10px 0 0 0;'/>", unsafe_allow_html=True)

render_hero(hero_path, hero_height_px)

# ================== INPUT FORM (ORDER-CRITICAL) ==================
st.markdown("### Input Parameters (match training order)")

# Thứ tự CỘT ĐẦU VÀO PHẢI TRÙNG TRAINING:
# ['β', 'L/B', 'b/B', 'm_i', 'GSI', 'γB/σ_ci']
c1, c2 = st.columns(2, gap="large")
with c1:
    st.latex(r"\beta")
    beta = st.number_input("beta", label_visibility="collapsed", format="%.6f", key="beta")
    st.latex(r"\frac{L}{B}")
    L_over_B = st.number_input("L_over_B", label_visibility="collapsed", format="%.6f", key="L_B")
    st.latex(r"\frac{b}{B}")
    b_over_B = st.number_input("b_over_B", label_visibility="collapsed", format="%.6f", key="b_B")
with c2:
    st.latex(r"m_i")
    m_i = st.number_input("m_i", label_visibility="collapsed", format="%.6f", key="mi")
    st.latex(r"\mathit{GSI}")
    GSI = st.number_input("GSI", label_visibility="collapsed", format="%.6f", key="GSI")
    st.latex(r"\frac{\gamma B}{\sigma_{ci}}")
    gammaB_over_sigci = st.number_input("gammaB_over_sigci", label_visibility="collapsed", format="%.6f", key="gB_sigci")

# ================== SESSION STATE ==================
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=["β","L/B","b/B","m_i","GSI","γB/σ_ci","N"])

# ================== PREDICT ==================
col_pred, col_clear = st.columns([1,1])
with col_pred:
    do_pred = st.button("⚛️ Predict")
with col_clear:
    if st.button("🧹 Clear runs"):
        st.session_state.results = st.session_state.results.iloc[0:0]  # clear all
        st.rerun()

if do_pred:
    try:
        cols = ["β","L/B","b/B","m_i","GSI","γB/σ_ci"]
        dfX = pd.DataFrame([[beta, L_over_B, b_over_B, m_i, GSI, gammaB_over_sigci]], columns=cols)

        # Scale X
        X_in = dfX.values
        X_scaled = scalerX.transform(X_in) if scalerX is not None else X_in

        # Predict
        y_pred_norm = model.predict(X_scaled)
        if scalerY is not None:
            y_pred = scalerY.inverse_transform(np.array(y_pred_norm).reshape(-1,1)).ravel()
        else:
            y_pred = np.array(y_pred_norm).ravel()

        N_val = float(y_pred[0])

        st.markdown("### Prediction Result")
        st.latex(rf"N: \; {N_val:.6f}")

        new_row = {
            "β":beta, "L/B":L_over_B, "b/B":b_over_B, "m_i":m_i,
            "GSI":GSI, "γB/σ_ci":gammaB_over_sigci, "N":N_val
        }
        st.session_state.results = pd.concat(
            [st.session_state.results, pd.DataFrame([new_row])],
            ignore_index=True
        )
    except Exception as e:
        st.error(f"Errors: {e}")

# ================== RESULTS TABLE ==================
def render_results_table_white(df: pd.DataFrame):
    rename_map = {"β":"β","L/B":"L/B","b/B":"b/B","m_i":"mᵢ","GSI":"GSI","γB/σ_ci":"γB/σci","N":"N"}
    df_show = df.rename(columns=rename_map).copy()

    df_fmt = df_show.copy()
    if "N" in df_fmt.columns:
        df_fmt["N"] = pd.to_numeric(df_fmt["N"], errors="coerce").map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    for col in ["β","L/B","b/B","mᵢ","GSI","γB/σci"]:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda v: (f"{v:.10g}" if isinstance(v,(int,float,np.floating)) else str(v)))

    styled = (df_fmt.style.hide(axis="index")
              .set_table_styles([
                  {"selector":"table","props":"border-collapse:collapse;width:100%;background:#fff;color:#000;font-size:15px;"},
                  {"selector":"th","props":"background:#fff;color:#000;text-align:center;font-weight:700;font-size:16px;padding:10px 12px;border-bottom:2px solid rgba(0,0,0,.25);"},
                  {"selector":"td","props":"background:#fff;color:#000;text-align:center;padding:10px 12px;border-bottom:1px solid rgba(0,0,0,.12);"},
              ]))
    st.markdown(styled.to_html(), unsafe_allow_html=True)

if not st.session_state.results.empty:
    st.markdown("### Table of Results")
    render_results_table_white(st.session_state.results)

    csv_bytes = st.session_state.results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_bytes, "prediction_results.csv", "text/csv")

    # ================== CHART ==================
    st.markdown("### Chart")

    # Thêm cột index để tiện vẽ theo thứ tự chạy
    df_plot = st.session_state.results.copy().reset_index().rename(columns={"index": "Run #"})
    x_options = ["Run #", "β", "L/B", "b/B", "m_i", "GSI", "γB/σ_ci"]
    point_size = st.slider("Point size", 30, 200, 80)
    x_choice = st.selectbox("X-axis", x_options, index=0, help="Chọn biến để vẽ theo trục X")

    # Vẽ scatter + line nối (theo thứ tự chạy)
    base = alt.Chart(df_plot).encode(
        x=alt.X(f"{x_choice}:Q" if x_choice != "Run #" else "Run #:Q", title=x_choice),
        y=alt.Y("N:Q", title="N"),
        tooltip=["Run #","β","L/B","b/B","m_i","GSI","γB/σ_ci","N"]
    )

    line = base.mark_line(opacity=0.5)
    pts  = base.mark_circle(size=point_size)

    st.altair_chart((line + pts).interactive(), use_container_width=True)
