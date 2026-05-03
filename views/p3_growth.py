import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statistics_analysis import get_growth_metrics
from views.helpers import CT, ct, kpi, card_open, card_close, fmt, pct_color

def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Growth Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:24px;">Deep-dive performance metrics and channel trajectory</div>', unsafe_allow_html=True)

    gm = get_growth_metrics(df)

    # Growth KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi"><div class="kpi-lbl">Weekly Growth</div><div class="kpi-val">{pct_color(gm["weekly_pct"])}</div><div class="kpi-sub">{fmt(gm["weekly_views"])} views this week</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi"><div class="kpi-lbl">Monthly Growth</div><div class="kpi-val">{pct_color(gm["monthly_pct"])}</div><div class="kpi-sub">{fmt(gm["monthly_views"])} views this month</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi"><div class="kpi-lbl">Yearly Growth</div><div class="kpi-val">{pct_color(gm["yearly_pct"])}</div><div class="kpi-sub">{fmt(gm["yearly_views"])} views this year</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(kpi("Avg Engagement Rate", f"{gm['avg_engagement']}%", "Likes + Comments / Views"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    import pandas as pd
    mdf = pd.DataFrame(gm["monthly_data"])
    if not mdf.empty:
        # Views over time line chart
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card_open("Views Over Time (Monthly)", "Total views per month"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mdf["month"], y=mdf["views"], fill="tozeroy",
                fillcolor="rgba(255,0,0,0.07)", line=dict(color="#ff0000",width=2.5), mode="lines+markers",
                marker=dict(size=4,color="#ff0000")))
            fig.update_layout(**ct(height=240, showlegend=False, xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=9))))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(card_close, unsafe_allow_html=True)

        with c2:
            st.markdown(card_open("Monthly Uploads (Bar Chart)", "Number of videos uploaded per month"), unsafe_allow_html=True)
            fig = px.bar(mdf, x="month", y="videos", color_discrete_sequence=["#ff0000"])
            fig.update_layout(**ct(height=240, showlegend=False, xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=9))))
            fig.update_traces(opacity=0.85)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(card_close, unsafe_allow_html=True)

        # Engagement over time
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(card_open("Engagement Rate Over Time", "Monthly average engagement %"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mdf["month"], y=mdf["engagement_rate"],
                line=dict(color="#ff4444",width=2), mode="lines+markers",
                marker=dict(size=4,color="#ff4444"), fill="tozeroy",
                fillcolor="rgba(255,68,68,0.06)"))
            fig.update_layout(**ct(height=230, showlegend=False,
                yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444",ticksuffix="%")))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(card_close, unsafe_allow_html=True)

        with c4:
            st.markdown(card_open("Views vs Likes (Dual Axis)", "Correlation between views and likes"), unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mdf["month"], y=mdf["views"], name="Views",
                marker_color="#ff0000", opacity=0.7, yaxis="y"))
            fig.add_trace(go.Scatter(x=mdf["month"], y=mdf["likes"], name="Likes",
                line=dict(color="#ff8888",width=2), mode="lines", yaxis="y2"))
            fig.update_layout(**ct(height=230, showlegend=True,
                legend=dict(font=dict(color="#666",size=10),bgcolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444",title="Views"),
                yaxis2=dict(overlaying="y",side="right",showgrid=False,color="#666",title="Likes")))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(card_close, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
