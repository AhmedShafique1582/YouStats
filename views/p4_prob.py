import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statistics_analysis import get_probability_analysis
from views.helpers import CT, ct, kpi, card_open, card_close, fmt

def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Probability Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:24px;">Statistical probability and Z-score analysis</div>', unsafe_allow_html=True)

    pa = get_probability_analysis(df)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi("P(Video > 1M Views)", f"{pa['p_1m']}%", f"Z = {pa['z_1m']}"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("P(Video > 10M Views)", f"{pa['p_10m']}%", f"Z = {pa['z_10m']}"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("Mean Views", fmt(pa['mean']), "Channel average"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("Std Deviation", fmt(pa['std']), "Views spread"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(card_open("Normal Distribution Curve",
            f"μ = {fmt(pa['mean'])} · σ = {fmt(pa['std'])}"), unsafe_allow_html=True)
        x = pa["curve_x"]; y = pa["curve_y"]
        mean_v = pa["mean"]; std_v = pa["std"]

        # Normalize Y to 0-1 so axis shows 0–1 instead of tiny scientific values
        y_norm = [v / max(y) for v in y]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_norm, line=dict(color="#ff0000", width=2.5),
            showlegend=False, hovertemplate="Views: %{x:.3s}<br>Density: %{y:.2f}<extra></extra>"))
        # Shade ±1σ
        mask1 = [mean_v - std_v <= xi <= mean_v + std_v for xi in x]
        x1s = [xi for xi, m in zip(x, mask1) if m]
        y1s = [yi for yi, m in zip(y_norm, mask1) if m]
        if x1s:
            fig.add_trace(go.Scatter(x=x1s + x1s[::-1], y=y1s + [0]*len(y1s),
                fill="toself", fillcolor="rgba(255,0,0,0.12)",
                line=dict(width=0), showlegend=True, name="±1σ (68% of videos)"))
        # Shade ±2σ
        mask2 = [mean_v - 2*std_v <= xi <= mean_v + 2*std_v for xi in x]
        x2s = [xi for xi, m in zip(x, mask2) if m]
        y2s = [yi for yi, m in zip(y_norm, mask2) if m]
        if x2s:
            fig.add_trace(go.Scatter(x=x2s + x2s[::-1], y=y2s + [0]*len(y2s),
                fill="toself", fillcolor="rgba(255,0,0,0.05)",
                line=dict(width=0), showlegend=True, name="±2σ (95% of videos)"))
        fig.add_vline(x=mean_v, line=dict(color="#888", width=1, dash="dash"))
        fig.add_annotation(x=mean_v, y=1.08, yref="paper",
            text=f"μ = {fmt(mean_v)}", showarrow=False, font=dict(color="#aaa", size=11))
        fig.update_layout(**ct(height=270, showlegend=True,
            legend=dict(font=dict(color="#666", size=9), bgcolor="rgba(0,0,0,0)", x=0, y=-0.15, orientation="h"),
            xaxis=dict(showgrid=False, color="#444",
                       title=dict(text="Views", font=dict(color="#555", size=10))),
            yaxis=dict(showgrid=False, visible=False),  # hide PDF density values
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c2:
        st.markdown(card_open("Histogram with Normal Curve Overlay",
            "Video count per view range vs fitted normal distribution"), unsafe_allow_html=True)

        import numpy as _np
        views_arr = df["views"].values
        n = len(views_arr)
        # Compute bins manually so we can scale the PDF to count space
        counts_h, edges_h = _np.histogram(views_arr, bins=35)
        bin_width = edges_h[1] - edges_h[0]
        bin_centers = (edges_h[:-1] + edges_h[1:]) / 2

        # Scale PDF → counts: pdf(x) * n * bin_width
        from scipy.stats import norm as _norm
        mu_h, sigma_h = float(pa["mean"]), float(pa["std"])
        pdf_scaled = _norm.pdf(bin_centers, mu_h, sigma_h) * n * bin_width

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_centers, y=counts_h, name="Videos",
            marker_color="#ff0000", opacity=0.65,
            hovertemplate="Views: %{x:.2s}<br>Videos: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers, y=pdf_scaled,
            line=dict(color="#fff", width=1.5, dash="dot"),
            name="Normal fit", mode="lines",
            hovertemplate="Views: %{x:.2s}<br>Expected: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(**ct(height=270, showlegend=True, bargap=0.04,
            legend=dict(font=dict(color="#666", size=10), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, color="#444",
                       title=dict(text="Views", font=dict(color="#555", size=10))),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444",
                       title=dict(text="Number of Videos", font=dict(color="#555", size=10))),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)



    c3, c4 = st.columns(2)

    with c3:
        st.markdown(card_open("Z-Score Bar Chart", "Top 20 videos by Z-score"), unsafe_allow_html=True)
        import pandas as pd
        zdf = pd.DataFrame(pa["z_scores"])
        if not zdf.empty:
            zdf["short"] = zdf["title"].str[:35]+"…"
            zdf["color"] = zdf["z_score"].apply(lambda v: "#ff0000" if v>0 else "#444")
            fig = go.Figure(go.Bar(x=zdf["z_score"], y=zdf["short"], orientation="h",
                marker_color=zdf["color"], opacity=0.85))
            fig.update_layout(**ct(height=290,
                yaxis=dict(showgrid=False,color="#555",linecolor="rgba(0,0,0,0)",tickfont=dict(size=9)),
                xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444")))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c4:
        st.markdown(card_open("Box Plot — Views Spread & Outliers"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Box(y=df["views"], name="Views", marker_color="#ff0000",
            boxmean="sd", jitter=0.3, pointpos=-1.8,
            boxpoints="outliers", marker_size=4))
        fig.update_layout(**ct(height=290, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
