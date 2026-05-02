import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from youtube_api import get_channel_id, get_channel_stats, get_all_videos
from statistics_analysis import (
    prepare_dataframe,
    get_descriptive_stats,
    get_confidence_intervals,
    fit_probability_distribution,
    build_regression_model,
    detect_outliers,
    categorize_videos
)

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="YouStat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=YouTube+Sans:wght@400;500;700&display=swap');

* {
    font-family: 'Roboto', sans-serif;
    box-sizing: border-box;
}

/* Main background - YouTube dark */
.stApp {
    background-color: #0f0f0f;
    color: #f1f1f1;
}

/* Sidebar - YouTube sidebar color */
[data-testid="stSidebar"] {
    background-color: #212121 !important;
    border-right: none !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div {
    padding-top: 16px;
}

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Radio buttons as YouTube nav items */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border-radius: 10px;
    padding: 10px 14px;
    cursor: pointer;
    color: #f1f1f1 !important;
    font-size: 14px;
    font-weight: 400;
    width: 100%;
    transition: background 0.1s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #3d3d3d !important;
}

[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: #3d3d3d !important;
    font-weight: 500;
}

/* Input field */
.stTextInput > div > div > input {
    background-color: #121212 !important;
    color: #f1f1f1 !important;
    border: 1px solid #303030 !important;
    border-radius: 40px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #1c62b9 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Analyze button */
.stButton > button {
    background-color: #ff0000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 20px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background-color: #cc0000 !important;
}

/* KPI Cards */
.yt-kpi {
    background: #212121;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #303030;
}

.yt-kpi-label {
    font-size: 12px;
    color: #aaaaaa;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}

.yt-kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f1f1;
    line-height: 1;
}

.yt-kpi-sub {
    font-size: 12px;
    color: #717171;
    margin-top: 6px;
}

/* Section cards */
.yt-section {
    background: #212121;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #303030;
    margin: 12px 0;
}

.yt-section-title {
    font-size: 16px;
    font-weight: 600;
    color: #f1f1f1;
    margin-bottom: 4px;
}

.yt-section-sub {
    font-size: 13px;
    color: #717171;
    margin-bottom: 20px;
}

/* Channel header */
.channel-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 20px 0 24px 0;
}

.channel-name {
    font-size: 24px;
    font-weight: 700;
    color: #f1f1f1;
    margin: 0;
}

.channel-sub-text {
    font-size: 13px;
    color: #aaaaaa;
    margin-top: 4px;
}

/* Divider */
.yt-divider {
    border: none;
    border-top: 1px solid #303030;
    margin: 16px 0;
}

/* Nav section label */
.nav-section-label {
    font-size: 11px;
    font-weight: 600;
    color: #717171;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 12px 14px 6px 14px;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #212121;
    border: 1px solid #303030;
    border-radius: 12px;
    padding: 16px;
}

/* Success/error messages */
.stSuccess {
    background: #1a2e1a !important;
    border: 1px solid #2d5a2d !important;
    border-radius: 8px !important;
    color: #4ade80 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #ff0000 !important;
}

/* Page title */
.page-title {
    font-size: 22px;
    font-weight: 700;
    color: #f1f1f1;
    margin: 0 0 4px 0;
}

.page-subtitle {
    font-size: 14px;
    color: #717171;
    margin-bottom: 20px;
}

/* Validation box */
.validation-box {
    background: #1a1a2e;
    border: 1px solid #303060;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}

.val-label {
    font-size: 11px;
    color: #717171;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}

.val-value {
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ──────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="#212121",
    plot_bgcolor="#212121",
    font=dict(color="#aaaaaa", family="Roboto", size=12),
    xaxis=dict(showgrid=False, color="#717171", linecolor="#303030"),
    yaxis=dict(showgrid=True, gridcolor="#303030", color="#717171", linecolor="#303030"),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#303030", font_color="#f1f1f1", bordercolor="#444"),
)

# ─── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 14px 8px 14px;">
        <span style="font-size:20px; font-weight:700; color:#f1f1f1;">
            <span style="color:#ff0000;">You</span>Stat
        </span>
        <span style="font-size:11px; color:#717171; margin-left:6px;">Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none; border-top:1px solid #303030; margin:8px 0">', unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Pages</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "Dashboard",
        "Video Insights",
        "Audience Analytics",
        "Growth Predictions"
    ], label_visibility="collapsed")

    st.markdown('<hr style="border:none; border-top:1px solid #303030; margin:8px 0">', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">Search</div>', unsafe_allow_html=True)

    channel_query = st.text_input("",
        placeholder="Channel name or ID",
        label_visibility="collapsed"
    )
    analyze_button = st.button("Analyze")

    st.markdown('<hr style="border:none; border-top:1px solid #303030; margin:8px 0">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px; color:#717171; padding: 0 14px;">YouStat © 2026</p>', unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ─── Fetch Data ────────────────────────────────────────
if analyze_button and channel_query:
    with st.spinner("Fetching channel data..."):
        channel_id = get_channel_id(channel_query)
        if not channel_id:
            st.error("Channel not found. Check the name or ID.")
            st.stop()
        channel_stats = get_channel_stats(channel_id)
        videos = get_all_videos(channel_stats["playlist_id"])
        df = prepare_dataframe(videos)
        st.session_state.data_loaded = True
        st.session_state.df = df
        st.session_state.channel_stats = channel_stats

# ─── Welcome Screen ────────────────────────────────────
if not st.session_state.data_loaded:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:70vh; text-align:center;">
        <div style="font-size:52px; font-weight:800; color:#f1f1f1; line-height:1.1; margin-bottom:16px;">
            <span style="color:#ff0000;">You</span>Stat
        </div>
        <div style="font-size:16px; color:#717171; max-width:400px; line-height:1.6;">
            YouTube channel analytics powered by real data.<br>
            Enter a channel name in the sidebar to begin.
        </div>
        <div style="margin-top:40px; display:flex; gap:32px;">
            <div style="text-align:center;">
                <div style="font-size:22px; font-weight:700; color:#f1f1f1;">Statistics</div>
                <div style="font-size:12px; color:#717171;">Descriptive & CI</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:22px; font-weight:700; color:#f1f1f1;">Distributions</div>
                <div style="font-size:12px; color:#717171;">Normal & Poisson</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:22px; font-weight:700; color:#f1f1f1;">Predictions</div>
                <div style="font-size:12px; color:#717171;">Regression Model</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Main App ──────────────────────────────────────────
else:
    df = st.session_state.df
    cs = st.session_state.channel_stats

    # ── DASHBOARD ──────────────────────────────────────
    if page == "Dashboard":

        # Channel header
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(cs["thumbnail"], width=80)
        with col2:
            st.markdown(f"""
            <div style="padding: 8px 0">
                <div class="channel-name">{cs['channel_name']}</div>
                <div class="channel-sub-text">
                    Joined {cs['created_at'][:10]} &nbsp;·&nbsp; {cs['total_videos']:,} videos
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">Subscribers</div>
                <div class="yt-kpi-value">{cs['subscribers']/1e6:.2f}M</div>
                <div class="yt-kpi-sub">Total subscribers</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">Total Views</div>
                <div class="yt-kpi-value">{cs['total_views']/1e9:.2f}B</div>
                <div class="yt-kpi-sub">All time views</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            avg_views = int(df["views"].mean())
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">Avg Views/Video</div>
                <div class="yt-kpi-value">{avg_views/1e6:.2f}M</div>
                <div class="yt-kpi-sub">Per video average</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            avg_eng = round(df["engagement_rate"].mean(), 2)
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">Avg Engagement</div>
                <div class="yt-kpi-value">{avg_eng}%</div>
                <div class="yt-kpi-sub">Likes + comments / views</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Growth chart
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="yt-section">', unsafe_allow_html=True)
            st.markdown('<div class="yt-section-title">Channel Growth</div>', unsafe_allow_html=True)
            st.markdown('<div class="yt-section-sub">Cumulative views over time</div>', unsafe_allow_html=True)
            fig = px.area(df, x="published_at", y="cumulative_views",
                         color_discrete_sequence=["#ff0000"])
            fig.update_traces(fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color="#ff0000", width=2))
            fig.update_layout(**CHART_THEME)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="yt-section">', unsafe_allow_html=True)
            st.markdown('<div class="yt-section-title">Views per Video</div>', unsafe_allow_html=True)
            st.markdown('<div class="yt-section-sub">Distribution of video views</div>', unsafe_allow_html=True)
            fig2 = px.histogram(df, x="views", nbins=40,
                               color_discrete_sequence=["#ff0000"])
            fig2.update_layout(**CHART_THEME)
            fig2.update_layout(showlegend=False,
                              bargap=0.1,
                              xaxis_title="Views",
                              yaxis_title="Videos")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recent videos
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Recent Videos</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Last 10 uploaded videos</div>', unsafe_allow_html=True)
        recent = df.sort_values("published_at", ascending=False).head(10)[
            ["title", "published_at", "views", "likes", "comments", "engagement_rate"]
        ].reset_index(drop=True)
        recent["published_at"] = recent["published_at"].dt.strftime("%Y-%m-%d")
        recent.columns = ["Title", "Date", "Views", "Likes", "Comments", "Engagement %"]
        st.dataframe(recent, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── VIDEO INSIGHTS ─────────────────────────────────
    elif page == "Video Insights":
        st.markdown('<div class="page-title">Video Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Deep dive into individual video performance</div>', unsafe_allow_html=True)

        # Descriptive stats
        desc = get_descriptive_stats(df)
        stats_df = pd.DataFrame(desc).T.reset_index()
        stats_df.columns = ["Metric"] + list(stats_df.columns[1:])

        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Mean, median, std dev and more across all variables</div>', unsafe_allow_html=True)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Box plots
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Outlier Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Box plots showing viral outliers vs normal videos</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        for col_name, container, color in zip(
            ["views", "likes", "comments"],
            [b1, b2, b3],
            ["#ff0000", "#ff6b6b", "#ff9999"]
        ):
            with container:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[col_name],
                    name=col_name.capitalize(),
                    marker_color=color,
                    boxmean=True,
                    line_color=color
                ))
                fig.update_layout(**CHART_THEME)
                fig.update_layout(
                    title=dict(text=col_name.capitalize(), font=dict(color="#f1f1f1", size=14)),
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Video categories
        df_cat, summary = categorize_videos(df)
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Video Performance Categories</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Viral, Hit, Average and Flop videos breakdown</div>', unsafe_allow_html=True)

        cat_counts = df_cat["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(cat_counts, x="Category", y="Count",
                        color="Category",
                        color_discrete_sequence=["#ff0000","#ff4444","#ff8888","#ffbbbb","#717171"])
        fig_cat.update_layout(**CHART_THEME)
        fig_cat.update_layout(showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

        st.dataframe(summary, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 10 videos
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Top 10 Videos</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Highest performing videos of all time</div>', unsafe_allow_html=True)
        top10 = df.nlargest(10, "views")[["title","published_at","views","likes","comments","engagement_rate"]]
        top10["published_at"] = top10["published_at"].dt.strftime("%Y-%m-%d")
        top10.columns = ["Title","Date","Views","Likes","Comments","Engagement %"]
        st.dataframe(top10, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── AUDIENCE ANALYTICS ─────────────────────────────
    elif page == "Audience Analytics":
        st.markdown('<div class="page-title">Audience Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Engagement patterns and statistical distributions</div>', unsafe_allow_html=True)

        # Confidence Intervals
        ci = get_confidence_intervals(df)
        ci_df = pd.DataFrame(ci).T.reset_index()
        ci_df.columns = ["Metric", "Mean", "Confidence Level", "CI Lower", "CI Upper", "Margin of Error"]

        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Confidence Intervals (95%)</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">95% confidence range for each metric</div>', unsafe_allow_html=True)
        st.dataframe(ci_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Engagement over time
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="yt-section">', unsafe_allow_html=True)
            st.markdown('<div class="yt-section-title">Engagement Rate Over Time</div>', unsafe_allow_html=True)
            fig = px.scatter(df, x="published_at", y="engagement_rate",
                           color_discrete_sequence=["#ff0000"],
                           opacity=0.6)
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(**CHART_THEME)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="yt-section">', unsafe_allow_html=True)
            st.markdown('<div class="yt-section-title">Upload Frequency</div>', unsafe_allow_html=True)
            upload_freq = df.groupby("month_year").size().reset_index(name="videos")
            upload_freq["month_year"] = upload_freq["month_year"].astype(str)
            fig2 = px.bar(upload_freq, x="month_year", y="videos",
                         color_discrete_sequence=["#ff0000"])
            fig2.update_layout(**CHART_THEME)
            fig2.update_layout(showlegend=False,
                              xaxis_title="Month", yaxis_title="Videos Uploaded")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Duration analysis
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Video Duration vs Views</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Does video length affect performance?</div>', unsafe_allow_html=True)
        df_dur = df[df["duration_minutes"] < 60]
        fig3 = px.scatter(df_dur, x="duration_minutes", y="views",
                         color="engagement_rate",
                         color_continuous_scale=["#303030","#ff0000"],
                         opacity=0.7,
                         hover_data=["title"])
        fig3.update_traces(marker=dict(size=6))
        fig3.update_layout(**CHART_THEME)
        fig3.update_layout(
            xaxis_title="Duration (minutes)",
            yaxis_title="Views",
            coloraxis_colorbar=dict(title="Engagement %", tickfont=dict(color="#aaa"))
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability distributions
        dist = fit_probability_distribution(df)
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Probability Distributions</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Normal & Poisson distribution fitting with KS test</div>', unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        for col_name, container in zip(["views", "likes", "comments"], [d1, d2, d3]):
            with container:
                p = dist[col_name]["normal"]["p_value"]
                fit = "Normal fit" if p > 0.05 else "Not normal"
                color = "#4ade80" if p > 0.05 else "#ff4444"
                st.markdown(f"""
                <div class="yt-kpi">
                    <div class="yt-kpi-label">{col_name.upper()}</div>
                    <div class="yt-kpi-value" style="font-size:16px">
                        μ = {dist[col_name]['normal']['mu']:,.0f}
                    </div>
                    <div class="yt-kpi-sub">σ = {dist[col_name]['normal']['sigma']:,.0f}</div>
                    <div style="margin-top:10px; font-size:12px; color:{color}; font-weight:500">
                        {fit} (p={p})
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── GROWTH PREDICTIONS ─────────────────────────────
    elif page == "Growth Predictions":
        st.markdown('<div class="page-title">Growth Predictions</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Linear regression model predicting future channel growth</div>', unsafe_allow_html=True)

        with st.spinner("Building model..."):
            result = build_regression_model(df)

        # Model metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">R² Score</div>
                <div class="yt-kpi-value">{result['r2_score']}</div>
                <div class="yt-kpi-sub">Model accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">Views Per Day</div>
                <div class="yt-kpi-value">{result['slope']/1e6:.2f}M</div>
                <div class="yt-kpi-sub">Regression slope</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="yt-kpi">
                <div class="yt-kpi-label">Mean Abs Error</div>
                <div class="yt-kpi-value">{result['mae']/1e9:.2f}B</div>
                <div class="yt-kpi-sub">Average prediction error</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main prediction chart
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">Predicted vs Actual Growth</div>', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-sub">Red = actual data · Blue = future prediction · Green = model fit on test data</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["days_since_start"],
            y=df["cumulative_views"],
            name="Actual Views",
            line=dict(color="#ff0000", width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.05)'
        ))
        fig.add_trace(go.Scatter(
            x=result["X_test"].flatten(),
            y=result["Y_pred"],
            name="Model Fit",
            line=dict(color="#4ade80", width=2, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=result["future_days"].flatten(),
            y=result["future_predictions"],
            name="Future Prediction",
            line=dict(color="#60a5fa", width=2, dash="dash")
        ))
        fig.update_layout(**CHART_THEME)
        fig.update_layout(
            height=420,
            xaxis_title="Days Since First Video",
            yaxis_title="Cumulative Views",
            legend=dict(bgcolor="#2a2a2a", bordercolor="#303030",
                       font=dict(color="#f1f1f1"))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Validation
        actual = cs["total_views"]
        predicted = result["future_predictions"][0]
        error_pct = abs(actual - predicted) / actual * 100

        st.markdown(f"""
        <div class="validation-box">
            <div class="yt-section-title" style="margin-bottom:20px">
                Model Validation — Today
            </div>
            <div style="display:flex; gap:48px; flex-wrap:wrap;">
                <div>
                    <div class="val-label">Predicted Views</div>
                    <div class="val-value" style="color:#60a5fa">{predicted:,.0f}</div>
                </div>
                <div>
                    <div class="val-label">Actual Views</div>
                    <div class="val-value" style="color:#ff0000">{actual:,}</div>
                </div>
                <div>
                    <div class="val-label">Prediction Error</div>
                    <div class="val-value" style="color:#4ade80">{error_pct:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Future predictions table
        st.markdown('<div class="yt-section">', unsafe_allow_html=True)
        st.markdown('<div class="yt-section-title">30 / 60 / 90 Day Forecast</div>', unsafe_allow_html=True)
        pred_30 = result["future_predictions"][30]
        pred_60 = result["future_predictions"][60]
        pred_90 = result["future_predictions"][90]
        pred_365 = result["future_predictions"][364]

        f1, f2, f3, f4 = st.columns(4)
        for container, days, val in zip(
            [f1, f2, f3, f4],
            ["30 Days", "60 Days", "90 Days", "365 Days"],
            [pred_30, pred_60, pred_90, pred_365]
        ):
            with container:
                st.markdown(f"""
                <div class="yt-kpi">
                    <div class="yt-kpi-label">+{days}</div>
                    <div class="yt-kpi-value" style="font-size:18px">{val/1e9:.3f}B</div>
                    <div class="yt-kpi-sub">Predicted total views</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)