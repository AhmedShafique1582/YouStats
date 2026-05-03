import plotly.graph_objects as go
import plotly.express as px
import numpy as np

CT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#666", size=11, family="Inter"),
    xaxis=dict(showgrid=False, color="#444", linecolor="rgba(255,255,255,0.05)", tickcolor="#444"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#444", linecolor="rgba(255,255,255,0.05)"),
    margin=dict(l=0, r=0, t=30, b=0),
    hoverlabel=dict(bgcolor="#1a1a1a", font_color="#fff", bordercolor="#333"),
)

def ct(**kwargs):
    d = CT.copy()
    d.update(kwargs)
    return d

def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="kpi-lbl">{label}</div><div class="kpi-val">{value}</div>{"<div class=kpi-sub>"+sub+"</div>" if sub else ""}</div>'

def card_open(title, sub=""):
    return f'<div class="card"><div class="card-title">{title}</div>{"<div class=card-sub>"+sub+"</div>" if sub else ""}'

card_close = '</div>'

def fmt(n):
    if n is None: return "—"
    n = float(n)
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return f"{n:,.0f}"

def pct_color(v):
    c = "#4ade80" if v >= 0 else "#f87171"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span style="color:{c};font-size:13px;font-weight:600;">{arrow} {abs(v):.1f}%</span>'
