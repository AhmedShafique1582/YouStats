import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from statistics_analysis import build_random_forest_model
from views.helpers import CT, ct, kpi, card_open, card_close, fmt

def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:24px;">Random Forest Regressor with log-transformed view counts</div>', unsafe_allow_html=True)

    with st.spinner("Training Random Forest model…"):
        rf = build_random_forest_model(df)

    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(kpi("R² Score", rf["r2"], "Model accuracy"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("MAE", fmt(rf["mae"]), "Mean Absolute Error"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("RMSE", fmt(rf["rmse"]), "Root Mean Sq. Error"), unsafe_allow_html=True)

    history = pd.DataFrame(rf["history"])
    future  = pd.DataFrame(rf["future"])
    split_i = rf["split_index"]

    # ── GRAPH 1: Actual Views History ──────────────────────────────────────────
    st.markdown(card_open("📈 Actual Views",
        "Real view counts for every published video"),
        unsafe_allow_html=True)

    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(
        x=history["index"], y=history["actual"], name="Actual Views",
        line=dict(color="#ff0000", width=2), mode="lines",
        fill="tozeroy", fillcolor="rgba(255,0,0,0.07)",
        hovertemplate="Video #%{x}<br>Actual: %{y:,.0f} views<extra></extra>",
    ))
    # Mark train/test split
    fig_actual.add_vline(x=split_i, line=dict(color="#555", width=1, dash="dash"))
    fig_actual.add_annotation(x=split_i, y=1.04, yref="paper", showarrow=False,
        text="Train | Test", font=dict(color="#555", size=10))

    fig_actual.update_layout(**ct(height=320, showlegend=False,
        xaxis=dict(showgrid=False, color="#444",
                   title=dict(text="Video Index", font=dict(color="#555", size=11))),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444",
                   title=dict(text="Views", font=dict(color="#555", size=11))),
    ))
    st.plotly_chart(fig_actual, use_container_width=True)
    st.markdown(card_close, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GRAPH 2: Predicted Views + Future Forecast ─────────────────────────────
    st.markdown(card_open("Predicted Views + Future Forecast (30 Videos)",
        "Model predictions for all historical videos · 🟢 next 30 future videos forecast"),
        unsafe_allow_html=True)

    fig_pred = go.Figure()

    # Predicted history line
    fig_pred.add_trace(go.Scatter(
        x=history["index"], y=history["predicted"], name="Predicted (History)",
        line=dict(color="rgba(180,180,180,0.85)", width=2), mode="lines",
        fill="tozeroy", fillcolor="rgba(180,180,180,0.05)",
        hovertemplate="Video #%{x}<br>Predicted: %{y:,.0f} views<extra></extra>",
    ))
    # Future ±20% confidence band
    fig_pred.add_trace(go.Scatter(
        x=list(future["index"]) + list(future["index"])[::-1],
        y=[v * 1.2 for v in future["predicted"]] + [v * 0.8 for v in future["predicted"]][::-1],
        fill="toself", fillcolor="rgba(74,222,128,0.08)",
        line=dict(width=0), showlegend=True, name="Forecast ±20% Band",
        hoverinfo="skip",
    ))
    # Future forecast line
    fig_pred.add_trace(go.Scatter(
        x=future["index"], y=future["predicted"], name="Future Forecast (30 Videos)",
        line=dict(color="#4ade80", width=2.5), mode="lines+markers",
        marker=dict(size=6, color="#4ade80", symbol="circle-open"),
        hovertemplate="Future Video #%{x}<br>Est. Views: %{y:,.0f}<extra>Forecast</extra>",
    ))
    # Divider between history and forecast
    fig_pred.add_vline(x=len(history), line=dict(color="#4ade80", width=1.5, dash="dash"))
    fig_pred.add_annotation(x=len(history), y=1.04, yref="paper", showarrow=False,
        text="→ Forecast Begins", font=dict(color="#4ade80", size=10))

    fig_pred.update_layout(**ct(height=320, showlegend=True,
        legend=dict(font=dict(color="#888", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", x=0, y=-0.18),
        xaxis=dict(showgrid=False, color="#444",
                   title=dict(text="Video Index", font=dict(color="#555", size=11))),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444",
                   title=dict(text="Views", font=dict(color="#555", size=11))),
    ))
    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown(card_close, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)



    # Feature importance
    fi = rf["feature_importance"]
    feat_names = {
        "duration_minutes":  "Duration (min)",
        "days_since_start":  "Days Since Start",
        "hour":              "Upload Hour",
        "day_of_week":       "Day of Week",
        "month_num":         "Month",
        "rolling_avg_5":     "Rolling Avg (5 videos)",
        "rolling_avg_10":    "Rolling Avg (10 videos)",
        "rolling_avg_20":    "Rolling Avg (20 videos)",
        "log_rolling_avg5":  "Log Rolling Avg",
        "days_since_last":   "Days Since Last Upload",
        "title_length":      "Title Length",
        "title_words":       "Title Word Count",
        "is_weekend":        "Is Weekend",
        "quarter":           "Quarter",
        "video_seq":         "Video Sequence #",
    }
    labels = [feat_names.get(k,k) for k in fi.keys()]
    values = list(fi.values())

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(card_open("Feature Importance", "What drives view predictions most"), unsafe_allow_html=True)
        # Dynamic red gradient for any number of features
        n_bars = len(values)
        bar_colors = [f"rgba(255,{max(0,int(30+i*160/max(n_bars-1,1)))},0,0.85)" for i in range(n_bars)]
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=bar_colors),
        ))
        fig.update_layout(**ct(height=230,
            yaxis=dict(showgrid=False,color="#888",tickfont=dict(size=10)),
            xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c4:
        st.markdown(card_open("Model Info", "Random Forest configuration & accuracy"), unsafe_allow_html=True)

        # Accuracy derived from R²
        accuracy_pct = round(rf["r2"] * 100, 1)
        acc_color = "#4ade80" if accuracy_pct >= 80 else "#facc15" if accuracy_pct >= 60 else "#f87171"
        acc_label = "Excellent" if accuracy_pct >= 85 else "Good" if accuracy_pct >= 70 else "Fair" if accuracy_pct >= 55 else "Weak"

        st.markdown(f"""
        <div style="text-align:center;padding:16px 0 20px;border-bottom:1px solid #1e1e1e;margin-bottom:14px;">
          <div style="font-size:11px;color:#444;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Model Accuracy</div>
          <div style="font-size:52px;font-weight:900;color:{acc_color};letter-spacing:-1px;line-height:1;">{accuracy_pct}%</div>
          <div style="font-size:11px;color:{acc_color};margin-top:4px;font-weight:600;">{acc_label}</div>
          <div style="height:5px;background:#1e1e1e;border-radius:3px;margin-top:12px;">
            <div style="width:{accuracy_pct}%;height:5px;background:{acc_color};border-radius:3px;transition:width .5s;"></div>
          </div>
          <div style="font-size:10px;color:#333;margin-top:6px;">Based on R² score = {rf['r2']}</div>
        </div>""", unsafe_allow_html=True)

        rows = [
            ("Algorithm",       "Random Forest"),
            ("Estimators",      "100 trees"),
            ("Target Transform","log1p(views)"),
            ("Train/Test Split","80% / 20%"),
            ("Features",        "5 variables"),
            ("MAE",             f"{fmt(rf['mae'])} views"),
            ("RMSE",            f"{fmt(rf['rmse'])} views"),
        ]
        for i, (label, value) in enumerate(rows):
            border = "border-bottom:1px solid #1e1e1e;" if i < len(rows)-1 else ""
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0;{border}">
              <span style="color:#555;font-size:12px;">{label}</span>
              <span style="color:#fff;font-size:13px;font-weight:600;">{value}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(card_close, unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)
