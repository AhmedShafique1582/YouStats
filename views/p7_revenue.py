import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from statistics_analysis import estimate_revenue
from views.helpers import CT, ct, kpi, card_open, card_close, fmt

def render(df, cs, rpm=4.0):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Revenue Estimation</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:13px;color:#555;margin-bottom:24px;">RPM-based revenue estimate (current RPM: <span style="color:#ff0000;font-weight:600;">${rpm}</span> per 1K views)</div>', unsafe_allow_html=True)

    rev = estimate_revenue(df, rpm)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi("Total Est. Revenue", f"${rev['total_revenue']:,.0f}", f"All-time @ ${rpm} RPM"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Avg Monthly Revenue", f"${rev['avg_monthly_revenue']:,.0f}", "Per month estimate"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("RPM Rate", f"${rpm}", "Revenue per 1K views"), unsafe_allow_html=True)
    with k4:
        total_views = cs.get("total_views", 0)
        st.markdown(kpi("Total Views", fmt(total_views), "Used for calculation"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    mrev = pd.DataFrame(rev["monthly_revenue"])
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(card_open("Revenue per Month (Bar Chart)", "Estimated earnings per month"), unsafe_allow_html=True)
        if not mrev.empty:
            fig = px.bar(mrev, x="month", y="revenue", color_discrete_sequence=["#ff0000"])
            fig.update_layout(**ct(height=250, showlegend=False,
                xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=9)),
                yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444",tickprefix="$")))
            fig.update_traces(opacity=0.85)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c2:
        st.markdown(card_open("Revenue Growth (Line Chart)", "Monthly revenue trend over time"), unsafe_allow_html=True)
        if not mrev.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mrev["month"], y=mrev["revenue"].cumsum(),
                fill="tozeroy", fillcolor="rgba(255,0,0,0.07)",
                line=dict(color="#ff0000",width=2.5), mode="lines"))
            fig.update_layout(**ct(height=250, showlegend=False,
                xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=9)),
                yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444",tickprefix="$")))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    cat_rev = pd.DataFrame(rev["category_revenue"])

    with c3:
        st.markdown(card_open("Revenue Share by Video Category", "Which category earns the most"), unsafe_allow_html=True)
        if not cat_rev.empty:
            fig = go.Figure(go.Pie(
                labels=cat_rev["category"], values=cat_rev["revenue"], hole=0.52,
                marker=dict(colors=["#ff0000","#cc0000","#ff4444","#ff8888"]),
                textfont=dict(size=11,color="#fff"),
            ))
            fig.update_layout(**ct(height=260, showlegend=True,
                legend=dict(font=dict(color="#666",size=10),bgcolor="rgba(0,0,0,0)",x=1)))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c4:
        st.markdown(card_open("Top 10 Revenue Videos"), unsafe_allow_html=True)
        top10 = pd.DataFrame(rev["top10_revenue"])
        if not top10.empty:
            top10["short"] = top10["title"].str[:38]+"…"
            top10["revenue_fmt"] = top10["revenue"].apply(lambda v: f"${v:,.0f}")
            st.markdown("<div style='margin-top:6px;'>", unsafe_allow_html=True)
            for _, row in top10.iterrows():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #1c1c1c;">
                  <div style="font-size:12px;color:#ccc;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{row['short']}</div>
                  <div style="font-size:13px;font-weight:700;color:#4ade80;">{row['revenue_fmt']}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(card_close, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
