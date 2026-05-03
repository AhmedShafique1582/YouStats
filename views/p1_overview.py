import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statistics_analysis import get_channel_health_score, estimate_revenue
from views.helpers import CT, ct, kpi, card_open, card_close, fmt

def render(df, cs):
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:20px;margin-bottom:28px;">
      <img src="{cs['thumbnail']}" style="width:72px;height:72px;border-radius:50%;border:2px solid #ff0000;box-shadow:0 0 18px rgba(255,0,0,0.25);">
      <div>
        <div style="font-size:26px;font-weight:800;color:#fff;">{cs['channel_name']}</div>
        <div style="font-size:12px;color:#555;margin-top:3px;">Joined {cs['created_at'][:10]} · {cs['total_videos']:,} videos published</div>
      </div>
    </div>""", unsafe_allow_html=True)

    rev = estimate_revenue(df)
    health = get_channel_health_score(df, cs)

    # KPI cards
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi("Subscribers", fmt(cs['subscribers']), "Total subscribers"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Total Views", fmt(cs['total_views']), "All-time views"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("Videos Published", f"{cs['total_videos']:,}", "Uploaded videos"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("Est. Revenue", f"${rev['total_revenue']:,.0f}", f"@ ${rev['rpm']} RPM"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(card_open("Monthly Views", "Total views per month — channel growth over time"), unsafe_allow_html=True)
        # Group by month for meaningful view of growth
        monthly = df.copy()
        monthly["month"] = monthly["published_at"].dt.to_period("M").astype(str)
        mdata = monthly.groupby("month").agg(views=("views","sum"), videos=("video_id","count")).reset_index()
        mdata = mdata.tail(36)  # last 3 years max for readability

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mdata["month"], y=mdata["views"],
            name="Views", marker_color="#ff0000", opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=mdata["month"], y=mdata["views"],
            mode="lines", line=dict(color="#ff4444", width=2),
            name="Trend", showlegend=False
        ))
        fig.update_layout(**ct(height=260, showlegend=False,
            xaxis=dict(showgrid=False, color="#444", tickfont=dict(size=9), tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444"),
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close, unsafe_allow_html=True)

    with c2:
        st.markdown(card_open("Channel Health Score", "Overall performance rating"), unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health['score'],
            domain={"x":[0,1],"y":[0,1]},
            title={"text": health['status'], "font":{"color":"#888","size":13}},
            number={"font":{"color":"#fff","size":42,"family":"Inter"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#333","tickfont":{"color":"#444","size":10}},
                "bar":{"color":"#ff0000","thickness":0.22},
                "bgcolor":"#1a1a1a",
                "bordercolor":"#222",
                "steps":[
                    {"range":[0,40],"color":"#2a1a1a"},
                    {"range":[40,70],"color":"#1f1a1a"},
                    {"range":[70,100],"color":"#1a1f1a"},
                ],
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", height=230, margin=dict(l=20,r=20,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Health factors progress bars
        for factor, score in health['factors'].items():
            pct = score / max(health['factors'].values()) * 100
            st.markdown(f"""
            <div style="margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;font-size:11px;color:#555;margin-bottom:3px;">
                <span>{factor}</span><span style="color:#888;">{score}pts</span>
              </div>
              <div style="height:3px;background:#1f1f1f;border-radius:2px;">
                <div style="width:{pct}%;height:3px;background:#ff0000;border-radius:2px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Scoring criteria detail
        avg_eng = round(df["engagement_rate"].mean(), 2)
        subs = cs.get("subscribers", 0)
        total_v = cs.get("total_videos", 0)
        total_views_cs = cs.get("total_views", 1)
        sub_ratio = round(subs / (total_views_cs + 1) * 100, 2)

        CRITERIA = [
            ("Engagement", 30, health['factors']['Engagement'],
             [("≥ 5%","30"), ("≥ 2%","20"), ("≥ 1%","10"), ("<1%","5")],
             f"Current: {avg_eng}%"),
            ("Consistency", 20, health['factors']['Consistency'],
             [("Very regular","20"), ("Moderate","15"), ("Irregular","10"), ("Sporadic","5")],
             "Based on upload variance"),
            ("Growth", 25, health['factors']['Growth'],
             [("≥ 50% growth","25"), ("≥ 20%","20"), ("> 0%","15"), ("Declining","5")],
             "Last 10 vs first 10 videos"),
            ("Volume", 15, health['factors']['Volume'],
             [("> 200 videos","15"), ("> 100","12"), ("> 50","8"), ("> 20","5"), ("≤ 20","3")],
             f"Current: {total_v} videos"),
            ("Subscriber Ratio", 10, health['factors']['Subscriber Ratio'],
             [("≥ 10%","10"), ("≥ 5%","8"), ("≥ 1%","5"), ("<1%","3")],
             f"Current: {sub_ratio}% (subs/views)"),
        ]

        st.markdown("""
        <div style="margin-top:18px;border-top:1px solid #1f1f1f;padding-top:14px;">
          <div style="font-size:10px;font-weight:700;color:#333;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Scoring Criteria</div>
        """, unsafe_allow_html=True)

        for factor, max_pts, earned, tiers, note in CRITERIA:
            tier_html = " · ".join([f'<span style="color:#555">{t[0]}</span> <span style="color:#888;font-weight:600">= {t[1]}pts</span>' for t in tiers])
            earned_color = "#4ade80" if earned >= max_pts * 0.8 else "#facc15" if earned >= max_pts * 0.5 else "#f87171"
            st.markdown(f"""
            <div style="margin-bottom:12px;padding:10px 12px;background:#161616;border-radius:10px;border:1px solid #1e1e1e;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                <span style="font-size:12px;font-weight:600;color:#ccc;">{factor}</span>
                <span style="font-size:11px;font-weight:700;color:{earned_color};">{earned} / {max_pts} pts</span>
              </div>
              <div style="font-size:10px;line-height:1.7;">{tier_html}</div>
              <div style="font-size:10px;color:#444;margin-top:4px;font-style:italic;">{note}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(card_close, unsafe_allow_html=True)


    # Video Category Breakdown (real data)
    from statistics_analysis import categorize_videos
    df_cat, cat_summary = categorize_videos(df.copy())
    cat_summary = cat_summary.reset_index()

    st.markdown(card_open("Video Category Breakdown", "Performance distribution based on view count relative to channel average"), unsafe_allow_html=True)
    c3, c4 = st.columns([1, 1])

    CAT_COLORS = {"Viral":"#ff0000","Hit":"#ff4444","Above Average":"#ff8888","Average":"#555","Flop":"#2a2a2a"}
    with c3:
        fig = go.Figure(go.Pie(
            labels=cat_summary["category"],
            values=cat_summary["video_count"],
            hole=0.55,
            marker=dict(colors=[CAT_COLORS.get(c,"#444") for c in cat_summary["category"]]),
            textfont=dict(size=11, color="#fff"),
        ))
        fig.update_layout(**ct(height=260, showlegend=True,
            legend=dict(font=dict(color="#666",size=10), bgcolor="rgba(0,0,0,0)", x=1.05)))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        for _, row in cat_summary.iterrows():
            color = CAT_COLORS.get(row["category"], "#444")
            pct = round(row["video_count"] / len(df) * 100, 1)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #1c1c1c;">
              <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>
                <span style="font-size:13px;color:#ccc;font-weight:500;">{row['category']}</span>
              </div>
              <div style="text-align:right;">
                <span style="font-size:14px;font-weight:700;color:#fff;">{int(row['video_count'])}</span>
                <span style="font-size:11px;color:#444;margin-left:5px;">videos · {pct}%</span>
              </div>
            </div>""", unsafe_allow_html=True)
        avg_views = int(df["views"].mean())
        st.markdown(f"""
        <div style="margin-top:14px;padding:10px 12px;background:#1a1a1a;border-radius:10px;border:1px solid #222;">
          <div style="font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;">Channel Avg Views</div>
          <div style="font-size:18px;font-weight:800;color:#fff;margin-top:4px;">{fmt(avg_views)}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(card_close, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
