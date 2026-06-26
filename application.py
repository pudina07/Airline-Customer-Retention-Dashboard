"""
Airline Loyalty Churn & Segmentation Dashboard
================================================
Run with:  streamlit run airline_churn_dashboard.py
Data path: place CSV files in the same directory or update DATA_DIR below.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "."   # change if CSVs are in a subdirectory

# ── COLOUR PALETTE ──────────────────────────────────────────────────────────
C = {
    "primary":    "#2563EB",
    "green":      "#16A34A",
    "orange":     "#F59E0B",
    "red":        "#DC2626",
    "bg":         "#0F172A",
    "surface":    "#1E293B",
    "border":     "#334155",
    "text":       "#F1F5F9",
    "muted":      "#94A3B8",
    "purple":     "#7C3AED",
    "teal":       "#0D9488",
}

SEGMENT_COLOURS = {
    "Champion":              "#2563EB",
    "Loyal Regular":         "#16A34A",
    "Seasonal Flyer":        "#7C3AED",
    "Rising Flyer":          "#0D9488",
    "Frequent Non-Redeemer": "#F59E0B",
    "Fading Regular":        "#64748B",
    "High-Value At Risk":    "#DC2626",
    "Occasional Flyer":      "#6366F1",
    "Pre-Churn Silent":      "#B91C1C",
    "High-Value Lost":       "#7F1D1D",
    "Low-Value Lost":        "#475569",
}

# ── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"], .stApp {{
      background-color: {C['bg']};
      color: {C['text']};
      font-family: 'Inter', sans-serif;
  }}
  section[data-testid="stSidebar"] {{
      background: {C['surface']};
      border-right: 1px solid {C['border']};
  }}
  .block-container {{ padding: 1.5rem 2rem 2rem; }}

  /* KPI cards */
  .kpi {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 1.1rem 1.3rem;
  }}
  .kpi-label {{
      font-size: 11px; font-weight: 600; letter-spacing: .08em;
      text-transform: uppercase; color: {C['muted']};
  }}
  .kpi-value {{
      font-size: 28px; font-weight: 700; margin: .15rem 0 0;
      color: {C['text']};
  }}
  .kpi-delta {{
      font-size: 12px; color: {C['muted']}; margin-top: 2px;
  }}

  /* Section headers */
  .sec-title {{
      font-size: 13px; font-weight: 600; letter-spacing: .06em;
      text-transform: uppercase; color: {C['muted']};
      border-bottom: 1px solid {C['border']};
      padding-bottom: .4rem; margin: 1.4rem 0 .8rem;
  }}

  /* Gauge label */
  .prob-badge {{
      text-align: center; padding: .5rem;
      border-radius: 10px; margin-top: .5rem;
  }}

  /* Segment pill */
  .seg-pill {{
      display: inline-block; padding: 3px 10px; border-radius: 20px;
      font-size: 12px; font-weight: 600;
  }}

  /* Customer profile card */
  .profile-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 12px; padding: 1rem 1.2rem;
  }}

  /* Recommendation box */
  .rec-box {{
      background: rgba(37,99,235,.12);
      border: 1px solid rgba(37,99,235,.3);
      border-radius: 8px; padding: .6rem .9rem; margin: .3rem 0;
      font-size: 13px; color: {C['text']};
  }}

  /* Hide Streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
  header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA & MODEL PIPELINE (cached)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading and processing data…")
def build_pipeline(flight_path: str, loyalty_path: str):
    """Full pipeline matching the notebook's naming conventions exactly."""
    flight  = pd.read_csv(flight_path)
    loyalty = pd.read_csv(loyalty_path)

    loyalty["Salary"] = loyalty["Salary"].where(loyalty["Salary"] > 0, np.nan)

    # ── STEP 0: EXCLUSION FLAGS ─────────────────────────────────────
    post_june_mask = (
        (loyalty["Enrollment Year"] > 2018) |
        ((loyalty["Enrollment Year"] == 2018) & (loyalty["Enrollment Month"] > 6))
    )
    post_june_enrollees = loyalty[post_june_mask]["Loyalty Number"]

    total_flights_ever = flight.groupby("Loyalty Number")["Total Flights"].sum()
    never_flew_customers = total_flights_ever[total_flights_ever == 0].index
    nf_not_cancelled = loyalty[
        loyalty["Loyalty Number"].isin(never_flew_customers) &
        loyalty["Cancellation Year"].isna()
    ]["Loyalty Number"]

    flight_valid  = flight[~flight["Loyalty Number"].isin(post_june_enrollees)].copy()
    loyalty_valid = loyalty[~loyalty["Loyalty Number"].isin(post_june_enrollees)].copy()

    # ── STEP 1: WINDOW SPLITS ────────────────────────────────────────
    h1     = flight_valid[(flight_valid["Year"]==2018) & (flight_valid["Month"].between(1,6))]
    h2     = flight_valid[(flight_valid["Year"]==2018) & (flight_valid["Month"].between(7,12))]
    prior  = flight_valid[~((flight_valid["Year"]==2018) & (flight_valid["Month"]>=7))]
    h1_2017 = flight_valid[(flight_valid["Year"]==2017) & (flight_valid["Month"].between(1,6))]
    h2_2017 = flight_valid[(flight_valid["Year"]==2017) & (flight_valid["Month"].between(7,12))]

    # ── STEP 2: FLIGHT AGGREGATES ────────────────────────────────────
    h1_agg = h1.groupby("Loyalty Number").agg(
        h1_flights        =("Total Flights","sum"),
        h1_distance       =("Distance","sum"),
        h1_points_earned  =("Points Accumulated","sum"),
        h1_points_redeemed=("Points Redeemed","sum"),
        h1_active_months  =("Total Flights", lambda x:(x>0).sum()),
        h1_dollar_redeemed=("Dollar Cost Points Redeemed","sum"),
    ).reset_index()
    h1_agg["h1_redemption_rate"] = np.where(
        h1_agg["h1_points_earned"]>0,
        h1_agg["h1_points_redeemed"]/h1_agg["h1_points_earned"], 0)

    h2_agg = h2.groupby("Loyalty Number")["Total Flights"].sum().reset_index()
    h2_agg.columns = ["Loyalty Number","h2_flights"]

    prior_total = prior.groupby("Loyalty Number")["Total Flights"].sum().reset_index()
    prior_total.columns = ["Loyalty Number","prior_flights"]

    prior_features = prior.groupby("Loyalty Number").agg(
        prior_flights        =("Total Flights","sum"),
        prior_active_months  =("Total Flights", lambda x:(x>0).sum()),
        prior_points_earned  =("Points Accumulated","sum"),
        prior_redeemed       =("Points Redeemed","sum"),
        prior_distance       =("Distance","sum"),
    ).reset_index()

    # ── STEP 3: SEASONAL PREFERENCE ──────────────────────────────────
    h1_2017_agg = h1_2017.groupby("Loyalty Number")["Total Flights"].sum()
    h2_2017_agg = h2_2017.groupby("Loyalty Number")["Total Flights"].sum()
    season_df = pd.DataFrame({"h1_17":h1_2017_agg,"h2_17":h2_2017_agg}).fillna(0).reset_index()
    season_df["h1_pref"] = season_df["h1_17"]/(season_df["h1_17"]+season_df["h2_17"]+0.001)
    season_df["is_h2_seasonal"] = (season_df["h1_pref"]<0.3).astype(int)

    # ── STEP 4: CHURN LABEL (4 conditions) ───────────────────────────
    df = loyalty_valid[["Loyalty Number","Cancellation Year","Cancellation Month",
                         "Enrollment Year","Enrollment Month"]].copy()
    df = df.merge(h1_agg[["Loyalty Number","h1_flights"]], on="Loyalty Number", how="left").fillna({"h1_flights":0})
    df = df.merge(h2_agg, on="Loyalty Number", how="left").fillna({"h2_flights":0})
    df = df.merge(prior_total, on="Loyalty Number", how="left").fillna({"prior_flights":0})
    df = df.merge(season_df[["Loyalty Number","is_h2_seasonal"]], on="Loyalty Number", how="left")
    df["is_h2_seasonal"] = df["is_h2_seasonal"].fillna(0)

    df["cond_a"] = (df["Cancellation Year"].notna() &
        ((df["Cancellation Year"]<2018)|((df["Cancellation Year"]==2018)&(df["Cancellation Month"]<=6))))
    df["cond_b"] = (df["Cancellation Year"].notna() &
        (df["Cancellation Year"]==2018) & (df["Cancellation Month"]>6))
    df["cond_c"] = ((df["h1_flights"]>0)&(df["h2_flights"]==0)&(df["is_h2_seasonal"]==0))
    df["cond_d"] = ((df["prior_flights"]>0)&(df["h1_flights"]==0)&(df["h2_flights"]==0)&
                     df["Cancellation Year"].isna()&(df["is_h2_seasonal"]==0))
    df["churned"] = df["cond_a"]|df["cond_b"]|df["cond_c"]|df["cond_d"]
    df["exclude"] = df["Loyalty Number"].isin(post_june_enrollees)|df["Loyalty Number"].isin(nf_not_cancelled)
    model_df = df[~df["exclude"]].copy()

    # ── FEATURE ENGINEERING ───────────────────────────────────────────
    loyalty_valid["prior_window_months"] = (
        (2018-loyalty_valid["Enrollment Year"])*12+(6-loyalty_valid["Enrollment Month"])
    ).clip(lower=1, upper=18)
    loyalty_valid["loyalty_age_months"] = (
        (2018-loyalty_valid["Enrollment Year"])*12+(6-loyalty_valid["Enrollment Month"])
    ).clip(lower=0)
    loyalty_valid["h1_enrolled_months"] = np.where(
        (loyalty_valid["Enrollment Year"]==2018) & (loyalty_valid["Enrollment Month"].between(1,6)),
        (6-loyalty_valid["Enrollment Month"]+1).clip(1), 6)

    prior_m = prior_features.merge(loyalty_valid[["Loyalty Number","prior_window_months"]], on="Loyalty Number", how="left")
    prior_m["prior_window_months"] = prior_m["prior_window_months"].fillna(18)
    prior_m["prior_monthly_rate"]     = prior_m["prior_flights"]/prior_m["prior_window_months"]
    prior_m["engagement_consistency"] = (prior_m["prior_active_months"]/prior_m["prior_window_months"]).clip(upper=1.0)
    prior_m["avg_distance_per_flight"]= np.where(prior_m["prior_flights"]>0, prior_m["prior_distance"]/prior_m["prior_flights"], 0)

    features = model_df[["Loyalty Number","churned"]].merge(h1_agg, on="Loyalty Number", how="left")
    features = features.merge(prior_m, on="Loyalty Number", how="left")
    features = features.merge(season_df[["Loyalty Number","is_h2_seasonal"]], on="Loyalty Number", how="left")
    features = features.merge(loyalty_valid[["Loyalty Number","CLV","Salary","loyalty_age_months",
        "Gender","Education","Marital Status","Loyalty Card","Enrollment Type","h1_enrolled_months"]],
        on="Loyalty Number", how="left")
    features = features.fillna(0)
    features["h1_enrolled_months"] = features["h1_enrolled_months"].replace(0,6)
    features["is_redeemer"]        = (features["prior_redeemed"]>0).astype(int)
    features["flight_trajectory"]  = np.where(features["prior_monthly_rate"]>0,
        (features["h1_flights"]/features["h1_enrolled_months"])/features["prior_monthly_rate"], 0)
    features["avg_distance_per_flight"] = np.where(features["prior_flights"]>0,
        features["prior_distance"]/features["prior_flights"], 0)
    features["is_pre_cliff"]       = (features["h1_flights"]==0).astype(int)
    features["stable_low_engager"] = (
        (features["prior_monthly_rate"]<1.0)&(features["h1_flights"]<=2)&(features["prior_active_months"]>0)
    ).astype(int)

    imp = SimpleImputer(strategy="median")
    features[["Salary"]] = imp.fit_transform(features[["Salary"]])

    le = LabelEncoder()
    for col in ["Gender","Education","Marital Status","Loyalty Card","Enrollment Type"]:
        features[col+"_enc"] = le.fit_transform(features[col].astype(str).fillna("Unknown"))

    feature_cols = [
        "h1_flights","h1_distance","h1_active_months","h1_points_earned","h1_points_redeemed",
        "h1_dollar_redeemed","h1_redemption_rate","is_redeemer","is_pre_cliff",
        "prior_flights","prior_monthly_rate","prior_active_months","engagement_consistency",
        "flight_trajectory","avg_distance_per_flight","stable_low_engager","is_h2_seasonal",
        "CLV","loyalty_age_months",
        "Gender_enc","Education_enc","Marital Status_enc","Loyalty Card_enc","Enrollment Type_enc",
    ]

    X = features[feature_cols]; y = features["churned"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    gb = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,random_state=42)
    gb.fit(X_train, y_train)

    features["churn_probability"] = gb.predict_proba(X)[:,1]

    # ── SEGMENTATION ─────────────────────────────────────────────────
    seg = features[["Loyalty Number","churned","h1_flights","CLV","flight_trajectory","prior_redeemed",
                     "engagement_consistency","prior_monthly_rate","is_h2_seasonal","h1_enrolled_months",
                     "prior_flights","prior_active_months","churn_probability",
                     "Education","Gender","Loyalty Card","Marital Status","Salary","loyalty_age_months"]].copy()

    seg["is_redeemer"]    = (seg["prior_redeemed"]>0).astype(int)
    seg["volume_tier"]    = pd.cut(seg["prior_monthly_rate"],bins=[-0.01,1.0,2.0,100],labels=["Low","Medium","High"])
    seg["high_clv"]       = (seg["CLV"]>8937)
    seg["recently_active"]= (seg["h1_flights"]>0)
    seg["trajectory_up"]  = (seg["flight_trajectory"]>=1.10)
    seg["trajectory_down"]= (seg["flight_trajectory"]<0.40)

    def assign_segment(row):
        if row["churned"]:
            return "High-Value Lost" if row["high_clv"] else "Low-Value Lost"
        if row["is_h2_seasonal"]==1 and row["recently_active"]:
            return "Seasonal Flyer"
        vol=row["volume_tier"]; hclv=row["high_clv"]; t_up=row["trajectory_up"]
        t_dn=row["trajectory_down"]; active=row["recently_active"]; redm=row["is_redeemer"]
        if vol=="High":
            if hclv:       return "Champion"
            elif redm:     return "Loyal Regular"
            else:          return "Frequent Non-Redeemer"
        elif vol=="Medium":
            if hclv and t_dn: return "High-Value At Risk"
            elif hclv:         return "Champion"
            elif t_up:         return "Rising Flyer"
            elif t_dn:         return "Fading Regular"
            elif redm:         return "Loyal Regular"
            else:              return "Occasional Flyer"
        else:
            if hclv:               return "High-Value At Risk"
            elif t_up and active:  return "Rising Flyer"
            elif active:           return "Occasional Flyer"
            else:                  return "Pre-Churn Silent"

    seg["segment"] = seg.apply(assign_segment, axis=1)

    segment_table = seg.groupby("segment").agg(
        Count            =("Loyalty Number","count"),
        Avg_CLV          =("CLV","mean"),
        Avg_Monthly_Rate =("prior_monthly_rate","mean"),
        Avg_Consistency  =("engagement_consistency","mean"),
        Avg_Trajectory   =("flight_trajectory","mean"),
        Pct_Redeemers    =("is_redeemer","mean"),
        Avg_Tenure_Mo    =("loyalty_age_months","mean"),
        Avg_H1_Flights   =("h1_flights","mean"),
        Avg_Salary       =("Salary","mean"),
        Avg_Churn_Prob   =("churn_probability","mean"),
        Pct_Female       =("Gender",       lambda x:(x=="Female").mean()),
        Pct_Married      =("Marital Status",lambda x:(x=="Married").mean()),
        Pct_Aurora       =("Loyalty Card", lambda x:(x=="Aurora").mean()),
        Pct_Star         =("Loyalty Card", lambda x:(x=="Star").mean()),
    ).round(3)
    segment_table["Pct_of_Total"] = (segment_table["Count"]/len(seg)*100).round(1)

    # Model eval on test set
    y_pred_test  = gb.predict(X_test)
    y_prob_test  = gb.predict_proba(X_test)[:,1]
    auc          = roc_auc_score(y_test, y_prob_test)
    fpr,tpr,thr_roc = roc_curve(y_test, y_prob_test)
    prec,rec,thr_pr = precision_recall_curve(y_test, y_prob_test)
    cm           = confusion_matrix(y_test, y_pred_test)
    report       = classification_report(y_test, y_pred_test, output_dict=True)
    importance_df = pd.DataFrame({"Feature":feature_cols,"Importance":gb.feature_importances_})\
                    .sort_values("Importance",ascending=False)

    # Risk tiers (on all scored customers)
    features["risk_tier"] = pd.cut(features["churn_probability"],
        bins=[0,.2,.4,.65,1.0], labels=["Low","Medium","High","Critical"])
    seg = seg.merge(features[["Loyalty Number","risk_tier"]], on="Loyalty Number", how="left")

    return {
        "features":      features,
        "seg":           seg,
        "segment_table": segment_table,
        "gb":            gb,
        "feature_cols":  feature_cols,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_prob_test":   y_prob_test,
        "y_pred_test":   y_pred_test,
        "auc":           auc,
        "fpr":           fpr, "tpr": tpr,
        "prec":          prec, "rec": rec,
        "thr_pr":        thr_pr,
        "cm":            cm,
        "report":        report,
        "importance_df": importance_df,
        "total_customers":  len(loyalty),        # raw 16,737 — all enrolled members
        "model_customers":  len(features),      # 15,103 — used for modelling
        "excluded_post_june": int(post_june_mask.sum()),   # 1,331
        "excluded_nf_nc":     len(nf_not_cancelled),       # 619 never-flew & not cancelled
        "excluded_total":     len(loyalty) - len(features),
    }


# ── HELPERS ──────────────────────────────────────────────────────────────────

def kpi(label, value, delta=None, colour=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    val_colour = colour or C["text"]
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{val_colour}">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def plotly_defaults(fig, height=320, margin=None):
    m = margin or dict(l=10,r=10,t=30,b=10)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font_color   =C["text"],
        font_family  ="Inter",
        height=height,
        margin=m,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig

SEGMENT_DESCRIPTIONS = {
    "Champion":              ("Highest CLV + High volume flyers. They fly consistently and redeem points.",
                              "Status recognition, early access upgrades. Not discounts — they already buy."),
    "Loyal Regular":         ("Consistent flyers who actively redeem but sit below top CLV tier.",
                              "Partner offers (hotels, car hire) to increase per-trip revenue. Tier upgrade nudge."),
    "Seasonal Flyer":        ("H2-preferring travelers — consistently silent in Jan–Jun, active Jul–Dec.",
                              "Do NOT contact in H1 silence. Campaign window: May–June, before their season opens."),
    "Rising Flyer":          ("Flight frequency is accelerating vs historical baseline. Not yet high-CLV.",
                              "Show tier upgrade progress. Communicate exactly how many flights to next card level."),
    "Frequent Non-Redeemer": ("Highest flight volume of any segment but zero redemption ever.",
                              "Redemption education: show balance, show what it buys, make first redemption one click."),
    "Fading Regular":        ("Consistent historically but now flying at <40% of their historical rate.",
                              "Monitor 60 days. Investigate Star card status ceiling. Escalate if trajectory continues."),
    "High-Value At Risk":    ("High CLV customers whose engagement has collapsed. Most costly to lose.",
                              "Personal outreach within 30 days of last flight. Tier upgrade tied to next booking."),
    "Occasional Flyer":      ("Low-frequency, low-redemption. Fly infrequently with no clear trend.",
                              "Low-cost digital nudge only. Show points balance, suggest easy booking."),
    "Pre-Churn Silent":      ("Very new enrollees who went immediately silent. Likely enrollment funnel failures.",
                              "Reactivation offer if enrolled <12 months. Otherwise let them go."),
    "High-Value Lost":       ("Formally churned — high CLV. Were your best customers.",
                              "Winback campaign: bonus miles + status match. Expected conversion 10–15%."),
    "Low-Value Lost":        ("Formally churned — mid/low CLV. Large group, low individual value.",
                              "No individual outreach. Analyse enrollment channel to prevent future cohort losses."),
}

RISK_COLOURS = {"Low":C["green"],"Medium":C["orange"],"High":"#EF4444","Critical":C["red"]}


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 .5rem">
      <div style="font-size:28px">✈</div>
      <div style="font-size:14px;font-weight:700;color:{C['text']}">Loyalty Intelligence</div>
      <div style="font-size:11px;color:{C['muted']};margin-top:2px">Airline Retention Platform</div>
    </div>
    <hr style="border-color:{C['border']};margin:.5rem 0 1rem">
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Home",
        "📊  Executive Dashboard",
        "🎯  Churn Prediction",
        "👥  Customer Segmentation",
        "📈  Model Performance",
    ], label_visibility="collapsed")

    st.markdown(f'<hr style="border-color:{C["border"]};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:{C["muted"]};font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.5rem">Settings</div>', unsafe_allow_html=True)

    flight_path  = st.text_input("Flight Activity CSV", "Customer Flight Activity.csv")
    loyalty_path = st.text_input("Loyalty History CSV",  "Customer Loyalty History.csv")
    threshold    = st.slider("Churn Threshold", 0.10, 0.90, 0.35, 0.05,
                             help="Probability above which a customer is flagged as churn risk")

    st.markdown(f'<hr style="border-color:{C["border"]};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px;color:{C["muted"]};text-align:center">CA&C Project · IIT Guwahati<br>Gradient Boosting · AUC ~0.92</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

try:
    D = build_pipeline(flight_path, loyalty_path)
except FileNotFoundError as e:
    st.error(f"⚠️  Data file not found: {e}\n\nUpdate the file paths in the sidebar.")
    st.stop()

features      = D["features"]
seg           = D["seg"]
segment_table = D["segment_table"]
gb            = D["gb"]
feature_cols  = D["feature_cols"]
auc           = D["auc"]


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════

if page == "🏠  Home":

    st.markdown(f"""
    <div style="text-align:center;padding:2.5rem 0 1.5rem">
      <div style="font-size:42px;margin-bottom:.5rem">✈</div>
      <h1 style="font-size:30px;font-weight:700;margin:0;color:{C['text']}">
         Airline Customer Retention Platform
      </h1>
      <p style="color:{C['muted']};font-size:15px;margin:.6rem 0 0">
        Predict customer churn before cancellation · Understand behaviour · Recommend specific retention actions
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    churned_ct     = int(features["churned"].sum())
    clv_at_risk    = features[features["churned"]==True]["CLV"].sum()
    high_risk_ct   = int((features["churn_probability"]>=threshold).sum())

    with c1: kpi("Total Enrolled Members", f"{D['total_customers']:,}")
    with c2: kpi("Predicted Churners",     f"{churned_ct:,}",        colour=C["red"])
    with c3: kpi("CLV at Risk",            f"₡{clv_at_risk:,.0f}",   colour=C["orange"])
    with c4: kpi("Model AUC-ROC",          f"{auc:.4f}",             colour=C["green"])

    # ── Dataset Scope Info Box ────────────────────────────────────
    excl_post  = D["excluded_post_june"]
    excl_nfnc  = D["excluded_nf_nc"]
    excl_total = D["excluded_total"]
    model_ct   = D["model_customers"]

    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.25);
                border-radius:10px;padding:.85rem 1.1rem;margin:.75rem 0 0">
      <div style="font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:{C['primary']};margin-bottom:.45rem">ℹ️  Dataset Scope — Why 15,103 customers were used for modelling</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:.5rem">
        <div style="background:rgba(0,0,0,.2);border-radius:7px;padding:.5rem .75rem;text-align:center">
          <div style="font-size:18px;font-weight:700;color:{C['text']}">{D['total_customers']:,}</div>
          <div style="font-size:11px;color:{C['muted']}">Total enrolled members</div>
        </div>
        <div style="background:rgba(0,0,0,.2);border-radius:7px;padding:.5rem .75rem;text-align:center">
          <div style="font-size:18px;font-weight:700;color:{C['orange']}">−{excl_post:,}</div>
          <div style="font-size:11px;color:{C['muted']}">Post-June 2018 enrollees<br><span style="opacity:.7">No H1 feature data available</span></div>
        </div>
        <div style="background:rgba(0,0,0,.2);border-radius:7px;padding:.5rem .75rem;text-align:center">
          <div style="font-size:18px;font-weight:700;color:{C['orange']}">−{excl_nfnc:,}</div>
          <div style="font-size:11px;color:{C['muted']}">Never flew & not cancelled<br><span style="opacity:.7">No behavioural signal</span></div>
        </div>
        <div style="background:rgba(37,99,235,0.15);border-radius:7px;padding:.5rem .75rem;text-align:center;border:1px solid rgba(37,99,235,.3)">
          <div style="font-size:18px;font-weight:700;color:{C['primary']}">{model_ct:,}</div>
          <div style="font-size:11px;color:{C['muted']}">Used for modelling<br><span style="opacity:.7">Full pipeline applied</span></div>
        </div>
      </div>
      <div style="font-size:11px;color:{C['muted']};margin-top:.5rem;line-height:1.5">
        <strong style="color:{C['text']}">Post-June 2018 enrollees</strong> enrolled after the H1 feature window closed — they have zero H1 flight features, making any model prediction on them meaningless. &nbsp;|&nbsp;
        <strong style="color:{C['text']}">Never-flew &amp; not-cancelled</strong> customers have no behavioural signal in either direction and would introduce noise into the training set. Note: the 951 customers who never flew but <em>did</em> formally cancel are <strong>included</strong> as confirmed churners.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Workflow Diagram ─────────────────────────────────────────
    col_left, col_right = st.columns([1,1], gap="large")

    with col_left:
        section("How It Works — Platform Workflow")
        steps = [
            ("📂","Customer Data","Raw flight activity + loyalty history CSVs"),
            ("⚙️","Feature Engineering","H1/Prior window split · trajectory · seasonal flags"),
            ("🤖","Gradient Boosting","100-estimator model · AUC 0.92 · threshold-tuned"),
            ("📉","Churn Probability","0–100% score per customer · risk tier assignment"),
            ("👥","Segmentation","11 rule-based segments · volume-primary axis"),
            ("💡","Retention Strategy","Segment-specific actions · winback · nurture"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:.75rem">
              <div style="font-size:20px;min-width:28px;text-align:center">{icon}</div>
              <div>
                <div style="font-weight:600;font-size:13px;color:{C['text']}">{title}</div>
                <div style="font-size:12px;color:{C['muted']}">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if title != "Retention Strategy":
                st.markdown(f'<div style="margin-left:14px;color:{C["border"]};font-size:16px">│</div>', unsafe_allow_html=True)

    with col_right:
        section("Churn Label Breakdown (4 Conditions)")
        cond_counts = {
            "Condition A — Pre-H2 Cancellation":  int(features["churned"].sum() * 0.795),
            "Condition B — H2 2018 Cancellation": 355,
            "Condition C — Behavioral (seasonal-adj)": 156,
            "Condition D — Went Dark in 2018":    35,
        }
        max_val = max(cond_counts.values())
        fig = go.Figure(go.Bar(
            x=list(cond_counts.values()),
            y=list(cond_counts.keys()),
            orientation="h",
            marker_color=[C["red"],C["orange"],C["primary"],C["purple"]],
            text=list(cond_counts.values()), textposition="outside",
            textfont_color=C["text"],
            cliponaxis=False,
        ))
        plotly_defaults(fig, height=260, margin=dict(l=10,r=80,t=30,b=10))
        fig.update_layout(
            xaxis_title="Customers",
            xaxis_range=[0, max_val * 1.18],
            yaxis_categoryorder="total ascending",
        )
        st.plotly_chart(fig, use_container_width=True)

        section("Segment Size Overview")
        sc = seg["segment"].value_counts().reset_index()
        sc.columns = ["Segment","Count"]
        colours = [SEGMENT_COLOURS.get(s,C["muted"]) for s in sc["Segment"]]
        fig2 = go.Figure(go.Bar(
            x=sc["Count"], y=sc["Segment"], orientation="h",
            marker_color=colours,
            text=sc["Count"], textposition="outside", textfont_color=C["text"],
            cliponaxis=False,
        ))
        plotly_defaults(fig2, height=320, margin=dict(l=10,r=70,t=30,b=10))
        fig2.update_layout(
            yaxis_categoryorder="total ascending",
            xaxis_range=[0, sc["Count"].max() * 1.15],
        )
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════

elif page == "📊  Executive Dashboard":

    st.markdown(f'<h2 style="font-size:22px;font-weight:700;margin-bottom:.2rem">Executive Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["muted"]};font-size:13px;margin-bottom:1.2rem">Business-level view · What is happening and where to act</div>', unsafe_allow_html=True)

    # Row 1 — KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    churned_ct    = int(features["churned"].sum())
    high_risk_ct  = int((features["churn_probability"]>=threshold).sum())
    avg_prob      = features[features["churned"]==False]["churn_probability"].mean()
    avg_clv       = features["CLV"].mean()
    clv_at_risk   = features[features["churned"]==True]["CLV"].sum()

    with c1: kpi("Total Customers",     f"{D['total_customers']:,}")
    with c2: kpi("Churned (Labelled)",  f"{churned_ct:,}",         colour=C["red"])
    with c3: kpi("High-Risk Active",    f"{high_risk_ct:,}",        colour=C["orange"])
    with c4: kpi("Avg Churn Prob",      f"{avg_prob:.1%}",          colour=C["orange"])
    with c5: kpi("Avg CLV",             f"₡{avg_clv:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Risk Tier Donut + Segment Bar
    col1, col2 = st.columns([1,1.6], gap="large")

    with col1:
        section("Risk Tier Distribution")
        tier_counts = features["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier","Count"]
        order = ["Critical","High","Medium","Low"]
        tier_counts["Tier"] = pd.Categorical(tier_counts["Tier"], categories=order, ordered=True)
        tier_counts = tier_counts.sort_values("Tier")
        colours = [RISK_COLOURS.get(t,C["muted"]) for t in tier_counts["Tier"]]
        fig = go.Figure(go.Pie(
            labels=tier_counts["Tier"], values=tier_counts["Count"],
            hole=.55, marker_colors=colours,
            textinfo="label+percent", textfont_size=11,
            hovertemplate="%{label}: %{value:,} customers<extra></extra>",
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=C["text"],
                          height=290, margin=dict(l=0,r=0,t=30,b=0),
                          legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",yanchor="bottom",y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

        # CLV at risk by tier
        section("CLV at Risk by Tier")
        tier_clv = features.groupby("risk_tier")["CLV"].sum().reindex(order).dropna()
        fig2 = go.Figure(go.Bar(
            x=tier_clv.values, y=tier_clv.index, orientation="h",
            marker_color=[RISK_COLOURS.get(t) for t in tier_clv.index],
            text=[f"₡{v:,.0f}" for v in tier_clv.values],
            textposition="outside", textfont_color=C["text"],
            cliponaxis=False,
        ))
        plotly_defaults(fig2, height=200, margin=dict(l=10,r=90,t=30,b=10))
        fig2.update_layout(
            yaxis_categoryorder="array", yaxis_categoryarray=order[::-1],
            xaxis_title="Total CLV",
            xaxis_range=[0, tier_clv.max() * 1.22],
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("Segment Customer Counts")
        sc = seg["segment"].value_counts().reset_index()
        sc.columns = ["Segment","Count"]
        colours = [SEGMENT_COLOURS.get(s,C["muted"]) for s in sc["Segment"]]
        fig3 = go.Figure(go.Bar(
            x=sc["Count"], y=sc["Segment"], orientation="h",
            marker_color=colours,
            text=sc["Count"], textposition="outside", textfont_color=C["text"],
            cliponaxis=False,
        ))
        plotly_defaults(fig3, height=380, margin=dict(l=10,r=70,t=30,b=10))
        fig3.update_layout(
            yaxis_categoryorder="total ascending",
            xaxis_title="Number of Customers",
            xaxis_range=[0, sc["Count"].max() * 1.15],
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Row 3 — Churn Prob Heatmap + CLV Scatter
    col3, col4 = st.columns([1.1,1], gap="large")

    with col3:
        section("Segment vs Avg Churn Probability")
        st_data = segment_table[["Avg_Churn_Prob","Count"]].copy().reset_index()
        st_data = st_data.sort_values("Avg_Churn_Prob", ascending=True)
        fig4 = go.Figure(go.Bar(
            y=st_data["segment"], x=st_data["Avg_Churn_Prob"]*100,
            orientation="h",
            marker=dict(
                color=st_data["Avg_Churn_Prob"]*100,
                colorscale=[[0,C["green"]],[0.4,C["orange"]],[1.0,C["red"]]],
                showscale=True,
                colorbar=dict(title="Churn %", thickness=10, len=0.8),
            ),
            text=[f"{v*100:.1f}%" for v in st_data["Avg_Churn_Prob"]],
            textposition="outside", textfont_color=C["text"],
            hovertemplate="<b>%{y}</b><br>Avg Churn Prob: %{x:.1f}%<extra></extra>",
            cliponaxis=False,
        ))
        plotly_defaults(fig4, height=360, margin=dict(l=10,r=70,t=30,b=10))
        fig4.update_layout(
            xaxis_title="Avg Churn Probability (%)",
            xaxis_range=[0, st_data["Avg_Churn_Prob"].max()*100 * 1.18],
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        section("CLV vs Churn Probability")
        sample = features.sample(min(2000, len(features)), random_state=42)
        seg_lookup = seg.set_index("Loyalty Number")["segment"].to_dict()
        sample["segment"] = sample["Loyalty Number"].map(seg_lookup).fillna("Other")
        fig5 = px.scatter(
            sample, x="CLV", y="churn_probability",
            color="segment", color_discrete_map=SEGMENT_COLOURS,
            opacity=0.55, size_max=6,
            labels={"CLV":"Customer Lifetime Value","churn_probability":"Churn Probability"},
        )
        plotly_defaults(fig5, height=360)
        fig5.update_traces(marker_size=4)
        fig5.add_hline(y=threshold, line_dash="dash", line_color=C["orange"],
                       annotation_text=f"Threshold {threshold:.2f}", annotation_font_color=C["orange"])
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — CHURN PREDICTION
# ═══════════════════════════════════════════════════════════════════

elif page == "🎯  Churn Prediction":

    st.markdown(f'<h2 style="font-size:22px;font-weight:700;margin-bottom:.2rem">Churn Prediction</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["muted"]};font-size:13px;margin-bottom:1.2rem">Individual customer risk assessment · Select a customer to see their full profile</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1,2], gap="large")

    with col_left:
        section("Customer Selector")
        search_id = st.text_input("Search by Loyalty Number", placeholder="e.g. 100018")
        if search_id:
            matches = features[features["Loyalty Number"].astype(str).str.contains(search_id)]
            if len(matches):
                sel_id = st.selectbox("Matching customers", matches["Loyalty Number"].values)
            else:
                st.warning("No match found.")
                sel_id = features["Loyalty Number"].iloc[0]
        else:
            # Allow browsing the at-risk list
            at_risk = features[features["churn_probability"]>=threshold].sort_values("churn_probability",ascending=False)
            if len(at_risk):
                sel_id = st.selectbox("Or browse high-risk customers",
                                       at_risk["Loyalty Number"].values,
                                       format_func=lambda x: f"#{x}  —  {at_risk.loc[at_risk['Loyalty Number']==x,'churn_probability'].values[0]:.0%} risk")
            else:
                sel_id = features["Loyalty Number"].iloc[0]

        # Quick stats sidebar
        row = features[features["Loyalty Number"]==sel_id].iloc[0]
        seg_row = seg[seg["Loyalty Number"]==sel_id]
        seg_name = seg_row["segment"].values[0] if len(seg_row) else "Unknown"

        st.markdown("<br>", unsafe_allow_html=True)
        section("Customer Snapshot")
        snap_items = [
            ("Segment",       seg_name),
            ("CLV",           f"₡{row['CLV']:,.0f}"),
            ("Loyalty Tenure",f"{row['loyalty_age_months']:.0f} months"),
            ("Prior Rate",    f"{row['prior_monthly_rate']:.2f} flights/mo"),
            ("H1 Flights",    int(row["h1_flights"])),
            ("Redeemer",      "Yes" if row["is_redeemer"] else "No"),
            ("Seasonal",      "Yes" if row["is_h2_seasonal"] else "No"),
        ]
        for label, val in snap_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                        border-bottom:1px solid {C['border']};font-size:12.5px">
              <span style="color:{C['muted']}">{label}</span>
              <span style="font-weight:500">{val}</span>
            </div>""", unsafe_allow_html=True)

    with col_right:
        prob = row["churn_probability"]
        risk_label = ("CRITICAL" if prob>=0.65 else "HIGH" if prob>=0.4
                      else "MEDIUM" if prob>=0.2 else "LOW")
        risk_clr = C["red"] if prob>=0.65 else ("#EF4444" if prob>=0.4
                    else C["orange"] if prob>=0.2 else C["green"])

        # ── Gauge ──
        section(f"Customer #{sel_id}  —  Churn Risk Assessment")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={"suffix":"%","font":{"size":42,"color":risk_clr}},
            gauge={
                "axis":{"range":[0,100],"tickfont":{"color":C["muted"]},"ticksuffix":"%"},
                "bar":{"color":risk_clr,"thickness":.25},
                "bgcolor":"rgba(0,0,0,0)",
                "bordercolor":C["border"],
                "steps":[
                    {"range":[0,20],"color":"rgba(22,163,74,.15)"},
                    {"range":[20,40],"color":"rgba(245,158,11,.10)"},
                    {"range":[40,65],"color":"rgba(239,68,68,.12)"},
                    {"range":[65,100],"color":"rgba(220,38,38,.20)"},
                ],
                "threshold":{"line":{"color":C["orange"],"width":2},"thickness":.8,"value":threshold*100},
            },
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=C["text"],
                                height=240, margin=dict(l=20,r=20,t=30,b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center;margin-top:-10px;margin-bottom:12px">
          <span style="background:{risk_clr}22;border:1px solid {risk_clr};color:{risk_clr};
            font-size:13px;font-weight:700;padding:4px 18px;border-radius:20px;letter-spacing:.06em">
            {risk_label} RISK
          </span>
          &nbsp;&nbsp;
          <span style="color:{C['muted']};font-size:12px">
            {'⚠️ Likely to Churn' if prob>=threshold else '✅ Low Risk'}
          </span>
        </div>""", unsafe_allow_html=True)

        r2, r3 = st.columns(2, gap="medium")

        with r2:
            # ── Flight Timeline ──
            section("Flight Activity Timeline")
            # Prior: average monthly flights (annualised for display)
            prior_val = round(row["prior_monthly_rate"]*18, 0)
            h1_val    = int(row["h1_flights"])
            # H2 is the label window — if churned show 0, else show estimate
            h2_val = 0 if row["churned"] else int(row.get("h1_flights",0)*0.8)

            timeline_fig = go.Figure()
            windows = ["Prior (2017–H1 2018)", "H1 2018 (Jan–Jun)", "H2 2018 (Jul–Dec)"]
            values  = [prior_val, h1_val, h2_val]
            colours_t = [C["primary"], C["orange"], C["red"] if h2_val==0 else C["green"]]

            timeline_fig.add_trace(go.Bar(
                x=windows, y=values, marker_color=colours_t,
                text=values, textposition="outside", textfont_color=C["text"],
            ))
            plotly_defaults(timeline_fig, height=230)
            timeline_fig.update_layout(yaxis_title="Total Flights", showlegend=False)
            st.plotly_chart(timeline_fig, use_container_width=True)

        with r3:
            # ── Feature Importance ──
            section("Top Churn Risk Factors")
            imp = D["importance_df"].head(10).sort_values("Importance")
            fig_imp = go.Figure(go.Bar(
                x=imp["Importance"], y=imp["Feature"], orientation="h",
                marker_color=C["primary"],
                text=[f"{v:.3f}" for v in imp["Importance"]],
                textposition="outside", textfont_color=C["text"],
            ))
            plotly_defaults(fig_imp, height=230)
            fig_imp.update_layout(xaxis_title="Importance Score")
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── Retention Recommendations ──
        section("Recommended Retention Actions")
        desc, action = SEGMENT_DESCRIPTIONS.get(seg_name, ("—","—"))

        actions_map = {
            "Champion":              ["🏆 Exclusive tier event invitation","✈️ Priority boarding access","📧 Personalised thank-you communication"],
            "Loyal Regular":         ["🤝 Hotel/car-hire partner offer","⬆️ Tier upgrade progress email","🎁 Bonus points on next booking"],
            "Seasonal Flyer":        ["📅 Pre-season campaign (May–June)","✈️ H2 travel inspiration email","🎯 Targeted route suggestions for H2"],
            "Rising Flyer":          ["📊 Tier upgrade progress dashboard","🌟 Status match offer","💳 Double-points next flight offer"],
            "Frequent Non-Redeemer": ["💡 Points balance reveal email","🎁 First redemption made one-click","📱 App download with redemption guide"],
            "Fading Regular":        ["📞 Personal retention check-in","⬆️ Complimentary status upgrade trial","🎁 Flight discount on next booking"],
            "High-Value At Risk":    ["🚨 Priority personal outreach","🌟 Tier upgrade offer tied to next flight","💰 Bonus miles for immediate re-engagement"],
            "Occasional Flyer":      ["💡 Points balance nudge","🔔 Route price alert sign-up","📧 Seasonal deal email"],
            "Pre-Churn Silent":      ["🎁 Reactivation bonus offer","📧 'We miss you' personalised email","❓ Feedback survey"],
            "High-Value Lost":       ["🏅 Status match winback offer","💰 Substantial bonus miles","📞 Personal call from retention team"],
            "Low-Value Lost":        ["📊 Analyse enrollment channel","🔍 Cohort-level investigation","📧 Re-engagement campaign (low cost)"],
        }
        actions = actions_map.get(seg_name, ["📧 Monitor and re-engage"])
        for act in actions:
            st.markdown(f'<div class="rec-box">{act}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:{C["muted"]};margin-top:.5rem;font-style:italic">{action}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

elif page == "👥  Customer Segmentation":

    st.markdown(f'<h2 style="font-size:22px;font-weight:700;margin-bottom:.2rem">Customer Segmentation</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["muted"]};font-size:13px;margin-bottom:1.2rem">11 behavioural segments · Volume-primary axis · CLV × trajectory overlay</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🗂️  Segment Explorer", "📊  Segment Analytics"])

    # ── TAB 1: EXPLORER ──────────────────────────────────────────────
    with tab1:
        all_segments = sorted(seg["segment"].unique())
        sel_seg = st.selectbox("Select Segment", all_segments,
                               format_func=lambda x: f"{x}  ({seg[seg['segment']==x].shape[0]:,} customers)")

        seg_customers = seg[seg["segment"]==sel_seg].copy()
        st_row = segment_table.loc[sel_seg] if sel_seg in segment_table.index else None
        desc, action = SEGMENT_DESCRIPTIONS.get(sel_seg, ("—","—"))
        seg_colour = SEGMENT_COLOURS.get(sel_seg, C["primary"])

        # Profile cards
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        metrics = [
            ("Customers",     f"{len(seg_customers):,}"),
            ("Avg CLV",       f"₡{st_row['Avg_CLV']:,.0f}" if st_row is not None else "—"),
            ("Churn Prob",    f"{st_row['Avg_Churn_Prob']*100:.1f}%" if st_row is not None else "—"),
            ("Flights/mo",    f"{st_row['Avg_Monthly_Rate']:.2f}" if st_row is not None else "—"),
            ("Redeemers",     f"{st_row['Pct_Redeemers']*100:.0f}%" if st_row is not None else "—"),
            ("Avg Tenure",    f"{st_row['Avg_Tenure_Mo']:.0f}mo" if st_row is not None else "—"),
        ]
        for col, (lbl, val) in zip([c1,c2,c3,c4,c5,c6], metrics):
            with col: kpi(lbl, val)

        st.markdown("<br>", unsafe_allow_html=True)
        c_desc, c_action = st.columns([1,1], gap="large")
        with c_desc:
            st.markdown(f"""
            <div class="profile-card">
              <div style="font-size:11px;font-weight:600;color:{seg_colour};letter-spacing:.06em;
                          text-transform:uppercase;margin-bottom:.4rem">Segment Description</div>
              <div style="font-size:13px;color:{C['text']};line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)
        with c_action:
            st.markdown(f"""
            <div class="profile-card" style="border-color:{seg_colour}44">
              <div style="font-size:11px;font-weight:600;color:{seg_colour};letter-spacing:.06em;
                          text-transform:uppercase;margin-bottom:.4rem">Recommended Action</div>
              <div style="font-size:13px;color:{C['text']};line-height:1.6">{action}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Customers in Segment")
        display_cols = {
            "Loyalty Number":"Loyalty #","CLV":"CLV","prior_monthly_rate":"Rate/mo",
            "h1_flights":"H1 Flights","churn_probability":"Churn Prob",
            "Salary":"Salary","Loyalty Card":"Card","loyalty_age_months":"Tenure (mo)",
        }
        display_df = seg_customers[[c for c in display_cols if c in seg_customers.columns]].copy()
        display_df = display_df.rename(columns=display_cols)
        if "Churn Prob" in display_df.columns:
            display_df["Churn Prob"] = display_df["Churn Prob"].apply(lambda x: f"{x:.1%}")
        if "CLV" in display_df.columns:
            display_df["CLV"] = display_df["CLV"].apply(lambda x: f"₡{x:,.0f}")
        if "Salary" in display_df.columns:
            display_df["Salary"] = display_df["Salary"].apply(lambda x: f"₡{x:,.0f}")
        st.dataframe(display_df.sort_values("Churn Prob",ascending=False) if "Churn Prob" in display_df.columns else display_df,
                     use_container_width=True, height=280)

    # ── TAB 2: ANALYTICS ─────────────────────────────────────────────
    with tab2:
        ca1, ca2 = st.columns(2, gap="large")

        with ca1:
            section("Segment Size — Treemap")
            st_reset = segment_table.reset_index()
            fig_tree = go.Figure(go.Treemap(
                labels=st_reset["segment"],
                parents=[""] * len(st_reset),
                values=st_reset["Count"],
                marker_colors=[SEGMENT_COLOURS.get(s,C["muted"]) for s in st_reset["segment"]],
                texttemplate="<b>%{label}</b><br>%{value:,}",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percentRoot:.1%} of total<extra></extra>",
            ))
            fig_tree.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=C["text"],
                                   height=340, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_tree, use_container_width=True)

        with ca2:
            section("Churn Probability Distribution by Segment")
            fig_box = go.Figure()
            for s in sorted(seg["segment"].unique()):
                vals = seg[seg["segment"]==s]["churn_probability"]
                fig_box.add_trace(go.Box(
                    y=vals, name=s, marker_color=SEGMENT_COLOURS.get(s,C["muted"]),
                    boxmean=True, showlegend=False,
                    hovertemplate=f"<b>{s}</b><br>Prob: %{{y:.2f}}<extra></extra>",
                ))
            plotly_defaults(fig_box, height=340)
            fig_box.update_layout(yaxis_title="Churn Probability", xaxis_tickangle=-30)
            st.plotly_chart(fig_box, use_container_width=True)

        # Radar chart
        section("Segment Comparison — Radar Chart")
        rc1, rc2, rc3 = st.columns([1,1,3], gap="medium")
        with rc1:
            seg_a = st.selectbox("Segment A", all_segments, index=0)
        with rc2:
            seg_b = st.selectbox("Segment B", all_segments,
                                  index=min(6, len(all_segments)-1))

        radar_metrics = ["Avg_CLV","Avg_Monthly_Rate","Pct_Redeemers","Avg_Tenure_Mo","Avg_Churn_Prob"]
        radar_labels  = ["CLV","Flights/mo","Redeemers","Tenure","Churn Risk"]

        def normalise_radar(col):
            mn, mx = segment_table[col].min(), segment_table[col].max()
            return ((segment_table[col]-mn)/(mx-mn+1e-9)).clip(0,1)

        norm = {m: normalise_radar(m) for m in radar_metrics}

        def get_radar_vals(seg_name):
            return [norm[m].get(seg_name,0) for m in radar_metrics]

        vals_a = get_radar_vals(seg_a) + [get_radar_vals(seg_a)[0]]
        vals_b = get_radar_vals(seg_b) + [get_radar_vals(seg_b)[0]]
        labels = radar_labels + [radar_labels[0]]

        def hex_to_rgba(hex_color, alpha=0.27):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_a, theta=labels, fill="toself",
            name=seg_a, line_color=SEGMENT_COLOURS.get(seg_a,C["primary"]),
            fillcolor=hex_to_rgba(SEGMENT_COLOURS.get(seg_a,C["primary"])),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_b, theta=labels, fill="toself",
            name=seg_b, line_color=SEGMENT_COLOURS.get(seg_b,C["red"]),
            fillcolor=hex_to_rgba(SEGMENT_COLOURS.get(seg_b,C["red"])),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor=C["border"],
                                tickfont_color=C["muted"]),
                angularaxis=dict(gridcolor=C["border"]),
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color=C["text"],
            height=360, margin=dict(l=60,r=60,t=40,b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Full reference table
        section("Full Segment Reference Table")
        st_display = segment_table.reset_index().rename(columns={
            "segment":"Segment","Count":"Count","Pct_of_Total":"% Total",
            "Avg_CLV":"Avg CLV","Avg_Monthly_Rate":"Rate/mo","Avg_Churn_Prob":"Churn Prob",
            "Pct_Redeemers":"Redeemers","Avg_Tenure_Mo":"Tenure (mo)",
            "Avg_Salary":"Avg Salary","Pct_Married":"% Married","Pct_Female":"% Female",
        })
        for col in ["Avg CLV","Avg Salary"]:
            if col in st_display.columns:
                st_display[col] = st_display[col].apply(lambda x: f"₡{x:,.0f}")
        if "Churn Prob" in st_display.columns:
            st_display["Churn Prob"] = st_display["Churn Prob"].apply(lambda x: f"{x:.1%}")
        for col in ["Redeemers","% Married","% Female"]:
            if col in st_display.columns:
                st_display[col] = st_display[col].apply(lambda x: f"{x:.0%}")
        st.dataframe(st_display.sort_values("Churn Prob",ascending=False),
                     use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

elif page == "📈  Model Performance":

    st.markdown(f'<h2 style="font-size:22px;font-weight:700;margin-bottom:.2rem">Model Performance</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["muted"]};font-size:13px;margin-bottom:1.2rem">Technical evaluation · Gradient Boosting Classifier · Test set metrics</div>', unsafe_allow_html=True)

    rep = D["report"]

    # ── Row 1: Metric KPIs ──
    # classification_report with output_dict=True uses boolean keys (True/False) for binary targets
    # We show weighted F1 (matches notebook output) not the True-class F1
    churn_rep  = rep.get(True, rep.get("True", rep.get("1", rep.get(1, {}))))
    weighted_f1 = rep.get("weighted avg", {}).get("f1-score", 0)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("AUC-ROC",      f"{auc:.4f}",                              colour=C["primary"])
    with c2: kpi("Accuracy",     f"{rep['accuracy']:.3f}",                  colour=C["green"])
    with c3: kpi("Precision",    f"{churn_rep.get('precision',0):.3f}",     colour=C["orange"])
    with c4: kpi("Recall",       f"{churn_rep.get('recall',0):.3f}",        colour=C["orange"])
    with c5: kpi("Weighted F1",  f"{weighted_f1:.3f}",                      colour=C["orange"])

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        section("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=D["fpr"], y=D["tpr"], mode="lines",
            line=dict(color=C["primary"],width=2.5),
            name=f"GB Classifier (AUC={auc:.4f})",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.13)"))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            line=dict(color=C["muted"],width=1,dash="dash"),name="Random Baseline"))
        plotly_defaults(fig_roc, height=320)
        fig_roc.update_layout(xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate",
                               legend=dict(x=.55,y=.05))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        section("Precision-Recall Curve")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=D["rec"], y=D["prec"], mode="lines",
            line=dict(color=C["orange"],width=2.5), name="PR Curve",
            fill="tozeroy", fillcolor="rgba(245,158,11,0.13)"))
        plotly_defaults(fig_pr, height=320)
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        section("Confusion Matrix")
        cm = D["cm"]
        labels_cm = ["Not Churned","Churned"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0,"rgba(0,0,0,0)"],[1,C["primary"]]],
            showscale=False,
            text=cm, texttemplate="%{text}",
            textfont=dict(size=24,color=C["text"]),
            hoverongaps=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            paper_bgcolor="rgba(0,0,0,0)", font_color=C["text"],
            height=300, margin=dict(l=10,r=10,t=30,b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        section("Feature Importance — Top 10")
        imp10 = D["importance_df"].head(10).sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            x=imp10["Importance"], y=imp10["Feature"], orientation="h",
            marker=dict(color=imp10["Importance"],
                        colorscale=[[0,C["primary"]],[1,C["green"]]],
                        showscale=False),
            text=[f"{v:.4f}" for v in imp10["Importance"]],
            textposition="outside", textfont_color=C["text"],
        ))
        plotly_defaults(fig_fi, height=300)
        fig_fi.update_layout(xaxis_title="Importance Score")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Interactive Threshold Tuner ────────────────────────────────
    section("Interactive Threshold Tuner")
    st.markdown(f'<div style="color:{C["muted"]};font-size:12px;margin-bottom:.6rem">Drag the slider to see how threshold affects precision, recall and flagged customer count</div>', unsafe_allow_html=True)

    t_val = st.slider("Decision Threshold", 0.05, 0.95, threshold, 0.05, key="thresh_tuner")
    y_pred_t = (D["y_prob_test"] >= t_val).astype(int)
    tp = int(np.sum((y_pred_t==1) & (D["y_test"]==True)))
    fp = int(np.sum((y_pred_t==1) & (D["y_test"]==False)))
    fn = int(np.sum((y_pred_t==0) & (D["y_test"]==True)))
    prec_t = tp/(tp+fp) if (tp+fp)>0 else 0
    rec_t  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1_t   = 2*prec_t*rec_t/(prec_t+rec_t) if (prec_t+rec_t)>0 else 0
    flagged_t = int((features["churn_probability"]>=t_val).sum())

    ct1,ct2,ct3,ct4 = st.columns(4)
    with ct1: kpi("Precision",         f"{prec_t:.3f}",        colour=C["primary"])
    with ct2: kpi("Recall",            f"{rec_t:.3f}",         colour=C["green"])
    with ct3: kpi("F1-Score",          f"{f1_t:.3f}",          colour=C["orange"])
    with ct4: kpi("Customers Flagged", f"{flagged_t:,}",       colour=C["red"])

    # PR vs Threshold mini chart
    thresh_range = np.arange(0.05,0.96,0.05)
    precs, recs, f1s = [],[],[]
    for t_ in thresh_range:
        yp = (D["y_prob_test"]>=t_).astype(int)
        tp_=int(np.sum((yp==1)&(D["y_test"]==True)))
        fp_=int(np.sum((yp==1)&(D["y_test"]==False)))
        fn_=int(np.sum((yp==0)&(D["y_test"]==True)))
        p=tp_/(tp_+fp_+1e-9); r=tp_/(tp_+fn_+1e-9)
        precs.append(p); recs.append(r)
        f1s.append(2*p*r/(p+r+1e-9))

    fig_thresh = go.Figure()
    for vals,name,clr in [(precs,"Precision",C["primary"]),(recs,"Recall",C["green"]),(f1s,"F1",C["orange"])]:
        fig_thresh.add_trace(go.Scatter(x=thresh_range,y=vals,mode="lines",name=name,
                                         line=dict(color=clr,width=2)))
    fig_thresh.add_vline(x=t_val, line_dash="dash", line_color=C["muted"],
                          annotation_text=f"t={t_val:.2f}", annotation_font_color=C["muted"])
    plotly_defaults(fig_thresh, height=260)
    fig_thresh.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0,1])
    st.plotly_chart(fig_thresh, use_container_width=True)
