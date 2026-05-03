import streamlit as st
from youtube_api import get_channel_id, get_channel_stats, get_all_videos
from statistics_analysis import prepare_dataframe

st.set_page_config(page_title="YouStat", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box;}
.stApp{background:#0f0f0f;color:#e0e0e0;}
[data-testid="stSidebar"]{background:#111!important;border-right:1px solid rgba(255,255,255,0.07)!important;}
[data-testid="stSidebar"]>div{background:transparent!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stDecoration"]{display:none;}
[data-testid="stSidebar"] .stRadio>div{gap:0;}
[data-testid="stSidebar"] .stRadio label>div:first-child{width:0!important;min-width:0!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important;opacity:0!important;}
[data-testid="stSidebar"] .stRadio label{display:flex;align-items:center;border-left:3px solid transparent;padding:11px 20px!important;color:#555!important;font-size:13px!important;font-weight:500!important;transition:all .15s;}
[data-testid="stSidebar"] .stRadio label:hover{color:#aaa!important;border-left-color:#333!important;}
[data-testid="stSidebar"] .stRadio label[data-checked="true"]{color:#fff!important;border-left-color:#ff0000!important;background:rgba(255,0,0,0.05)!important;}
[data-testid="stSidebar"] .stRadio label p{color:inherit!important;font-size:inherit!important;font-weight:inherit!important;margin:0!important;}
.stButton>button{background:#e60000!important;color:#fff!important;border:none!important;border-radius:50px!important;font-weight:800!important;font-size:14px!important;letter-spacing:0.3px!important;padding:12px 22px!important;box-shadow:none!important;transition:all .2s ease!important;width:100%!important;}
.stButton>button:hover{background:#ff1a1a!important;transform:translateY(-2px)!important;box-shadow:none!important;}
.stButton>button:active{transform:translateY(0px)!important;}
[data-testid="stSpinner"]{display:flex!important;justify-content:center!important;align-items:center!important;}
[data-testid="stSpinner"]>div{display:flex!important;justify-content:center!important;}
[data-testid="stSpinner"] span{display:none!important;}
div.stSpinner>div>div{border-top-color:#ff0000!important;border-right-color:#ff0000!important;}
.stTextInput>div>div>input{background:#1e1e1e!important;border:1px solid rgba(255,255,255,0.18)!important;border-radius:10px!important;color:#ffffff!important;padding:10px 14px!important;caret-color:#ff0000!important;font-size:13px!important;}
.stTextInput>div>div>input::placeholder{color:rgba(255,255,255,0.3)!important;}
.stTextInput>div>div>input:focus{border-color:#ff0000!important;background:#252525!important;outline:none!important;box-shadow:0 0 0 2px rgba(255,0,0,0.12)!important;}
.stTextInput>div>div{background:transparent!important;}
.stTextInput label{color:#888!important;font-size:12px!important;}
.stNumberInput>div>div>input{background:#1e1e1e!important;border:1px solid rgba(255,255,255,0.18)!important;border-radius:10px!important;color:#ffffff!important;padding:10px 14px!important;caret-color:#ff0000!important;font-size:13px!important;}
.stNumberInput>div>div>input:focus{border-color:#ff0000!important;background:#252525!important;outline:none!important;box-shadow:0 0 0 2px rgba(255,0,0,0.12)!important;}
.stNumberInput>div>div>button{background:#1e1e1e!important;border-color:rgba(255,255,255,0.1)!important;color:#888!important;}
.stNumberInput>div>div>button:hover{background:#2a2a2a!important;color:#fff!important;}
[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}
[data-testid="metric-container"]{background:#1a1a1a;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:18px;}
.kpi{background:#1a1a1a;border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:20px 22px;margin:4px 0;}
.kpi-lbl{font-size:10px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;}
.kpi-val{font-size:28px;font-weight:800;color:#fff;}
.kpi-sub{font-size:11px;color:#444;margin-top:5px;}
.card{background:#1a1a1a;border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:22px;margin:6px 0;}
.card-title{font-size:15px;font-weight:700;color:#fff;margin-bottom:3px;}
.card-sub{font-size:12px;color:#555;margin-bottom:18px;}
.stSelectbox>div>div{background:#1a1a1a!important;border:1px solid rgba(255,255,255,0.1)!important;border-radius:10px!important;color:#fff!important;}
div[data-baseweb="select"]>div{background:#1a1a1a!important;border-color:rgba(255,255,255,0.1)!important;}
.stSlider>div>div>div{background:#ff0000!important;}
.stInfo{background:#1a1a2e!important;border:1px solid #303060!important;border-radius:10px!important;}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 20px 18px;border-bottom:1px solid #1c1c1c;">
      <div style="font-size:18px;font-weight:800;color:#fff;letter-spacing:-0.5px;">You<span style="color:#ff0000;">Stat</span></div>
      <div style="font-size:9px;color:#333;text-transform:uppercase;letter-spacing:2.5px;margin-top:3px;">Analytics Studio</div>
    </div>
    <div style="padding:16px 20px 6px;font-size:9px;font-weight:700;color:#333;text-transform:uppercase;letter-spacing:2px;">Pages</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Channel Overview",
        "Descriptive Statistics",
        "Growth Analysis",
        "Probability Analysis",
        "Prediction Model",
        "Best Time to Post",
        "Revenue Estimation",
    ], label_visibility="collapsed")

    st.markdown("""<div style="padding:16px 20px 6px;border-top:1px solid #1c1c1c;margin-top:10px;font-size:9px;font-weight:700;color:#333;text-transform:uppercase;letter-spacing:2px;">Search</div>""", unsafe_allow_html=True)

    channel_query = st.text_input("", placeholder="Channel name or ID", label_visibility="collapsed")
    st.markdown('<div style="padding:8px 20px 4px;font-size:9px;font-weight:700;color:#333;text-transform:uppercase;letter-spacing:2px;">RPM ($/1K views)</div>', unsafe_allow_html=True)
    rpm_val = st.number_input("", min_value=0.1, max_value=50.0, value=4.0, step=0.5,
                              format="%.1f", label_visibility="collapsed",
                              help="Revenue Per Mille — average earnings per 1,000 views")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    analyze = st.button("Analyze Channel  ›")

    st.markdown("""<div style="padding:14px 20px;margin-top:6px;font-size:10px;color:#252525;">YouStat © 2026</div>""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for k in ["loaded","df","cs"]:
    if k not in st.session_state:
        st.session_state[k] = False if k=="loaded" else None

# ── Fetch ─────────────────────────────────────────────────────────────────────
if analyze and channel_query:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;">
        <div style="width:60px;height:60px;border:4px solid rgba(255,0,0,0.1);border-top-color:#ff0000;border-radius:50%;animation:spin 1s linear infinite;"></div>
        <style>@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}</style>
        <div style="margin-top:24px;font-size:13px;color:#888;letter-spacing:2px;font-weight:700;text-transform:uppercase;animation:pulse 1.5s ease-in-out infinite;">Fetching & Analyzing Data...</div>
        <style>@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.5;}}</style>
    </div>
    """, unsafe_allow_html=True)
    
    cid = get_channel_id(channel_query)
    if not cid:
        loading_placeholder.empty()
        st.error("Channel not found."); st.stop()
    
    cs = get_channel_stats(cid)
    videos = get_all_videos(cs["playlist_id"])
    df = prepare_dataframe(videos)
    
    st.session_state.loaded = True
    st.session_state.df = df
    st.session_state.cs = cs
    
    loading_placeholder.empty()
    st.rerun()

# ── Routing ───────────────────────────────────────────────────────────────────
if not st.session_state.loaded:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:100vh;margin-top:-80px;text-align:center;">
      <div style="font-size:72px;font-weight:900;letter-spacing:-3px;color:#fff;margin-bottom:10px;">You<span style="color:#ff0000;">Stat</span></div>
      <div style="font-size:16px;color:#333;">Advanced YouTube analytics — powered by real data.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df = st.session_state.df
cs = st.session_state.cs

if page == "Channel Overview":
    from views.p1_overview import render; render(df, cs)
elif page == "Descriptive Statistics":
    from views.p2_stats import render; render(df, cs)
elif page == "Growth Analysis":
    from views.p3_growth import render; render(df, cs)
elif page == "Probability Analysis":
    from views.p4_prob import render; render(df, cs)
elif page == "Prediction Model":
    from views.p5_predict import render; render(df, cs)
elif page == "Best Time to Post":
    from views.p6_timing import render; render(df, cs)
elif page == "Revenue Estimation":
    from views.p7_revenue import render; render(df, cs, rpm_val)