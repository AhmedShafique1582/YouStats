import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats
from views.helpers import CT, ct, kpi, card_open, card_close, fmt


def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Views vs Likes — Scatter Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:24px;">Relationship between video views and audience engagement (likes)</div>', unsafe_allow_html=True)

    # ── Correlation Stats ────────────────────────────────────────────────────────
    corr, p_val = stats.pearsonr(df["views"], df["likes"])
    slope, intercept, r_val, _, _ = stats.linregress(df["views"], df["likes"])
    avg_lpv = df["likes_per_view"].mean() * 100  # as percentage

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi("Pearson Correlation", f"{corr:.4f}", "Views ↔ Likes strength"), unsafe_allow_html=True)
    with k2:
        strength = "Very Strong" if abs(corr) > 0.8 else "Strong" if abs(corr) > 0.6 else "Moderate" if abs(corr) > 0.4 else "Weak"
        st.markdown(kpi("Correlation Strength", strength, f"R² = {r_val**2:.4f}"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi("Avg Likes / View", f"{avg_lpv:.2f}%", "Likes per 100 views"), unsafe_allow_html=True)
    with k4:
        p_label = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        sig = "Significant ✓" if p_val < 0.05 else "Not Significant"
        st.markdown(kpi("P-Value", p_label, sig), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Color Coding Options ─────────────────────────────────────────────────────
    col_opt, _ = st.columns([2, 3])
    with col_opt:
        color_by = st.selectbox(
            "Color points by:",
            ["Video Category", "Year", "Day of Week", "Duration Bucket"],
            key="scatter_color_by"
        )

    # Prepare color column
    df2 = df.copy()
    if color_by == "Video Category":
        mean_v = df2["views"].mean(); std_v = df2["views"].std()
        def cat(v):
            if v > mean_v + 2 * std_v: return "Viral"
            elif v > mean_v + std_v:   return "Hit"
            elif v > mean_v:           return "Above Average"
            elif v > mean_v - std_v:   return "Average"
            else:                      return "Flop"
        df2["color_col"] = df2["views"].apply(cat)
        color_map = {"Viral": "#ff0000", "Hit": "#ff6600", "Above Average": "#ffaa00", "Average": "#888888", "Flop": "#444444"}
    elif color_by == "Year":
        df2["color_col"] = df2["year"].astype(str)
        color_map = None
    elif color_by == "Day of Week":
        df2["color_col"] = df2["day_name"]
        color_map = None
    else:  # Duration Bucket
        def dur_bucket(d):
            if d < 5:    return "< 5 min"
            elif d < 15: return "5–15 min"
            elif d < 30: return "15–30 min"
            else:        return "> 30 min"
        df2["color_col"] = df2["duration_minutes"].apply(dur_bucket)
        color_map = None

    df2["short_title"] = df2["title"].str[:55] + "…"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Scatter Plot ────────────────────────────────────────────────────────
    st.markdown(card_open("Views vs Likes — Interactive Scatter Plot", "Each dot = one video. Hover for details. Regression line shows trend."), unsafe_allow_html=True)

    # Build regression line
    x_line = np.linspace(df["views"].min(), df["views"].max(), 200)
    y_line = slope * x_line + intercept

    fig = px.scatter(
        df2, x="views", y="likes",
        color="color_col",
        color_discrete_map=color_map,
        hover_data={"short_title": True, "views": True, "likes": True,
                    "engagement_rate": True, "color_col": False},
        labels={"color_col": color_by, "views": "Views", "likes": "Likes",
                "short_title": "Title", "engagement_rate": "Eng. Rate (%)"},
        opacity=0.75,
    )

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name="Trend Line",
        line=dict(color="#ff0000", width=2, dash="dash"),
        opacity=0.8,
    ))

    fig.update_layout(**ct(
        height=480,
        showlegend=True,
        legend=dict(font=dict(color="#666", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                   color="#444", title=dict(text="Views", font=dict(color="#666", size=12))),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                   color="#444", title=dict(text="Likes", font=dict(color="#666", size=12))),
        annotations=[dict(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text=f"r = {corr:.3f} | R² = {r_val**2:.3f}",
            showarrow=False,
            font=dict(color="#888", size=12),
            bgcolor="rgba(26,26,26,0.8)",
            bordercolor="#333",
            borderwidth=1,
            borderpad=6,
        )],
    ))
    fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_close, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Likes/View Ratio ─────────────────────────────────────────────────────────
    st.markdown(card_open("Likes-per-View Ratio — Top 20 Videos", "Higher ratio = more engaged audience per view"), unsafe_allow_html=True)
    top_lpv = df2.nlargest(20, "likes_per_view")[["short_title", "likes_per_view", "views", "likes"]].copy()
    top_lpv["lpv_pct"] = (top_lpv["likes_per_view"] * 100).round(2)
    fig2 = px.bar(
        top_lpv, x="lpv_pct", y="short_title", orientation="h",
        color="lpv_pct",
        color_continuous_scale=[[0, "#330000"], [0.5, "#cc0000"], [1, "#ff4444"]],
        labels={"lpv_pct": "Likes / 100 Views", "short_title": ""},
        hover_data={"views": True, "likes": True, "lpv_pct": True},
    )
    fig2.update_layout(**ct(
        height=420,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(showgrid=False, color="#888", tickfont=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444"),
    ))
    fig2.update_traces(opacity=0.85)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(card_close, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
