import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statistics_analysis import get_descriptive_stats, get_confidence_intervals
from views.helpers import CT, ct, kpi, card_open, card_close, fmt

def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:24px;">Statistical analysis of video performance metrics</div>', unsafe_allow_html=True)

    desc = get_descriptive_stats(df)
    ci   = get_confidence_intervals(df)
    views_ci = ci["views"]

    # CI KPIs
    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(kpi("Mean Views", fmt(desc['views']['mean']), "Average per video"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("95% CI Lower", fmt(views_ci['ci_lower']), "Confidence lower bound"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("95% CI Upper", fmt(views_ci['ci_upper']), "Confidence upper bound"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats table
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown(card_open("Statistical Metrics Summary", "Mean, Median, Mode, SD, Variance, Skewness, Kurtosis"), unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for stat in ["mean","median","mode","std_dev","variance","skewness","kurtosis"]:
            rows.append({"Metric": stat.replace("_"," ").title(),
                         "Views": f"{desc['views'][stat]:,.2f}",
                         "Likes": f"{desc['likes'][stat]:,.2f}",
                         "Comments": f"{desc['comments'][stat]:,.2f}",
                         "Eng. Rate": f"{desc['engagement_rate'][stat]:,.2f}"})
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c2:
        st.markdown(card_open("95% Confidence Intervals", "Population parameter estimates"), unsafe_allow_html=True)
        for col in ["views","likes","comments","engagement_rate"]:
            c = ci[col]
            label = col.replace("_"," ").title()
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              <div style="font-size:11px;color:#555;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;">{label}</div>
              <div style="font-size:13px;color:#fff;font-weight:700;">{fmt(c['mean'])} <span style="color:#555;font-size:11px;font-weight:400;">mean</span></div>
              <div style="font-size:11px;color:#666;">[ {fmt(c['ci_lower'])} → {fmt(c['ci_upper'])} ] <span style="color:#ff0000;">±{fmt(c['margin_of_error'])}</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown(card_close, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(card_open("Views Distribution (Histogram)", "Frequency of video view counts"), unsafe_allow_html=True)
        fig = px.histogram(df, x="views", nbins=40, color_discrete_sequence=["#ff0000"])
        fig.update_layout(**ct(height=230, bargap=0.05))
        fig.update_traces(opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c4:
        st.markdown(card_open("Box Plot — Outlier Detection", "Views & Likes spread with outliers"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Box(y=df["views"], name="Views", marker_color="#ff0000", boxmean=True))
        fig.add_trace(go.Box(y=df["likes"], name="Likes", marker_color="#cc0000", boxmean=True))
        fig.update_layout(**ct(height=230, showlegend=True,
            legend=dict(font=dict(color="#666",size=10),bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    st.markdown(card_open("Top 10 Most Viewed Videos", "Highest performing videos of all time"), unsafe_allow_html=True)
    top10 = df.nlargest(10,"views")[["title","views","likes","comments","engagement_rate"]].reset_index(drop=True)
    top10["short"] = top10["title"].str[:50] + "…"
    fig = px.bar(top10, x="views", y="short", orientation="h",
                 color_discrete_sequence=["#ff0000"],
                 hover_data={"likes": True, "comments": True, "engagement_rate": True, "short": False})
    fig.update_layout(**ct(height=340,
        yaxis=dict(showgrid=False, color="#888", linecolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444")))
    fig.update_traces(opacity=0.85)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_close, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
