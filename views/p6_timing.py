import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from statistics_analysis import get_best_time_to_post
from views.helpers import CT, ct, kpi, card_open, card_close, fmt

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTHS_SHORT = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:26px;font-weight:800;color:#fff;margin-bottom:4px;">Best Time to Post</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:24px;">Optimal upload timing based on historical performance</div>', unsafe_allow_html=True)

    bt = get_best_time_to_post(df)

    k1,k2,k3,k4 = st.columns(4)
    hour_label = f"{bt['best_hour']:02d}:00"
    with k1: st.markdown(kpi("Best Day to Post", bt["best_day"], "Highest avg views"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Best Hour to Post", hour_label, "Highest avg views"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("Total Videos", f"{len(df):,}", "In dataset"), unsafe_allow_html=True)
    with k4:
        top_month_num = max(bt["monthly_avg"], key=bt["monthly_avg"].get, default="1")
        top_month = MONTHS_SHORT.get(int(top_month_num), "N/A")
        st.markdown(kpi("Best Month", top_month, "Highest avg views"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(card_open("Heatmap: Day × Hour Engagement", "Average views per day/hour combination"), unsafe_allow_html=True)
        hdf = pd.DataFrame(bt["heatmap"])
        if not hdf.empty:
            pivot = hdf.pivot_table(index="day", columns="hour", values="views", fill_value=0)
            pivot = pivot.reindex(range(7), fill_value=0)
            pivot.index = [DAYS[i] if i < len(DAYS) else str(i) for i in pivot.index]
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns],
                y=list(pivot.index),
                colorscale=[[0,"#1a1a1a"],[0.5,"#880000"],[1,"#ff0000"]],
                showscale=True, colorbar=dict(tickfont=dict(color="#555",size=9)),
            ))
            fig.update_layout(**ct(height=270, xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=9)),
                yaxis=dict(showgrid=False,color="#555",tickfont=dict(size=10))))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c2:
        st.markdown(card_open("Avg Views by Weekday", "Which day generates the most views"), unsafe_allow_html=True)
        day_vals = [bt["day_avg"].get(d, 0) for d in DAYS]
        colors = ["#ff0000" if d==bt["best_day"] else "#333" for d in DAYS]
        fig = go.Figure(go.Bar(x=DAYS, y=day_vals, marker_color=colors, opacity=0.85))
        fig.update_layout(**ct(height=270, showlegend=False,
            xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=10)),
            yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",color="#444")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(card_open("Monthly Performance (Radar)", "Average views by month"), unsafe_allow_html=True)
        month_labels = [MONTHS_SHORT[int(k)] for k in sorted(bt["monthly_avg"].keys())]
        month_values = [bt["monthly_avg"][k] for k in sorted(bt["monthly_avg"].keys())]
        if month_labels:
            fig = go.Figure(go.Scatterpolar(
                r=month_values + [month_values[0]],
                theta=month_labels + [month_labels[0]],
                fill="toself", fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="#ff0000",width=2),
                marker=dict(size=5,color="#ff0000"),
            ))
            fig.update_layout(polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True,color="#333",gridcolor="#222",tickfont=dict(size=8,color="#444")),
                angularaxis=dict(color="#666",gridcolor="#222",tickfont=dict(size=10,color="#666")),
            ), paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=40,r=40,t=30,b=30),
            font=dict(color="#666",family="Inter"))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c4:
        st.markdown(card_open("Upload Distribution by Day", "How uploads are spread across weekdays"), unsafe_allow_html=True)
        upload_vals = [bt["upload_by_day"].get(d, 0) for d in DAYS]
        fig = go.Figure(go.Pie(
            labels=DAYS, values=upload_vals, hole=0.5,
            marker=dict(colors=["#ff0000","#cc0000","#ff4444","#ff6666","#ff8888","#333","#444"]),
            textfont=dict(size=10,color="#fff"),
        ))
        fig.update_layout(**ct(height=260, showlegend=True,
            legend=dict(font=dict(color="#666",size=9),bgcolor="rgba(0,0,0,0)",x=1)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
