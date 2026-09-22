"""
charts.py — interactive Plotly charts.

Every chart shares one warm theme and is hoverable and zoomable. Returned figures
are rendered with st.plotly_chart(fig, use_container_width=True).
"""
import plotly.graph_objects as go
from constants import (COLOR_ATTENTION, COLOR_GRID, COLOR_MUTED, COLOR_RISK, COLOR_SURFACE,
                       COLOR_TEXT, COLOR_THRIVING, RISK_COLORS, RISK_LABELS)

BAND_LINES = [(33, COLOR_THRIVING, "Thriving ceiling"), (66, COLOR_ATTENTION, "Needs Attention ceiling")]


def _style(fig, height=340, title=None, showlegend=False):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Playfair Display, serif", size=17, color="#2d1a0e")) if title else None,
        paper_bgcolor=COLOR_SURFACE, plot_bgcolor=COLOR_SURFACE,
        font=dict(family="DM Sans, sans-serif", size=12, color=COLOR_TEXT),
        margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        showlegend=showlegend,
        hoverlabel=dict(bgcolor="white", bordercolor=COLOR_GRID,
                        font=dict(family="DM Sans, sans-serif", color=COLOR_TEXT)),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zeroline=False, linecolor=COLOR_GRID, tickfont=dict(color=COLOR_MUTED))
    fig.update_yaxes(gridcolor=COLOR_GRID, zeroline=False, linecolor=COLOR_GRID, tickfont=dict(color=COLOR_MUTED))
    return fig


def _bands(fig, xref_max=None):
    for y, color, name in BAND_LINES:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash="dash"), opacity=0.55,
                      annotation_text=name, annotation_position="top left",
                      annotation_font=dict(size=10, color=color))
    return fig


def score_history(dates, scores):
    """A student's score over time, with the band thresholds marked."""
    fig = go.Figure(go.Scatter(
        x=list(dates), y=list(scores), mode="lines+markers+text",
        line=dict(color=COLOR_RISK, width=3, shape="spline"),
        marker=dict(size=11, color="white", line=dict(color=COLOR_RISK, width=3)),
        text=[str(s) for s in scores], textposition="top center",
        textfont=dict(size=11, color="#2d1a0e"),
        fill="tozeroy", fillcolor="rgba(212,96,58,0.10)",
        hovertemplate="<b>%{y}/100</b><br>%{x|%d %b %Y}<extra></extra>"))
    _bands(fig)
    fig.update_yaxes(range=[0, 115], title_text="Score")
    return _style(fig, height=330, title="Your Wellness Journey")


def risk_pie(counts):
    """Share of students per risk level. counts: {level: n}."""
    levels = [l for l in ["Low", "Medium", "High"] if counts.get(l)]
    fig = go.Figure(go.Pie(
        labels=[RISK_LABELS[l] for l in levels], values=[counts[l] for l in levels],
        marker=dict(colors=[RISK_COLORS[l] for l in levels], line=dict(color="white", width=3)),
        hole=0.45, sort=False,
        textinfo="percent", textfont=dict(color="white", size=13),
        hovertemplate="<b>%{label}</b><br>%{value} students (%{percent})<extra></extra>"))
    return _style(fig, title="Students by Risk Level", showlegend=True)


def score_bars(labels, values, horizontal=False, title=None, axis_title="Average score"):
    """Average score per group, coloured by the band each average falls in."""
    colors = [COLOR_RISK if v > 66 else COLOR_ATTENTION if v > 33 else COLOR_THRIVING for v in values]
    text = [f"{v:.0f}" for v in values]
    if horizontal:
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                               text=text, textposition="outside",
                               hovertemplate="<b>%{y}</b><br>%{x:.0f}/100<extra></extra>"))
        fig.update_xaxes(range=[0, 115], title_text=axis_title)
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                               text=text, textposition="outside",
                               hovertemplate="<b>%{x}</b><br>%{y:.0f}/100<extra></extra>"))
        fig.update_yaxes(range=[0, 115], title_text=axis_title)
    fig.update_traces(textfont=dict(color=COLOR_TEXT, size=11), marker_line_width=0)
    return _style(fig, title=title)


def trend(dates, scores, counts=None):
    """Institution-wide average score over time."""
    hover = "<b>%{y:.0f}/100</b><br>%{x|%d %b %Y}"
    fig = go.Figure(go.Scatter(
        x=list(dates), y=list(scores), mode="lines+markers",
        line=dict(color=COLOR_RISK, width=3, shape="spline"),
        marker=dict(size=9, color="white", line=dict(color=COLOR_RISK, width=3)),
        fill="tozeroy", fillcolor="rgba(212,96,58,0.10)",
        customdata=list(counts) if counts is not None else None,
        hovertemplate=(hover + "<br>%{customdata} submissions<extra></extra>") if counts is not None
                      else (hover + "<extra></extra>")))
    _bands(fig)
    fig.update_yaxes(range=[0, 115], title_text="Average score")
    return _style(fig, height=330, title="Institution Wellness Trend")


def importance(labels, values):
    """Permutation importance — accuracy lost when an answer is scrambled."""
    colors = [COLOR_RISK if i < 2 else COLOR_ATTENTION if i < 5 else "#e8d5c0" for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h", marker_color=colors[::-1],
        hovertemplate="<b>%{y}</b><br>%{x:.1f} accuracy points<extra></extra>"))
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title_text="Accuracy lost if scrambled (pts)")
    return _style(fig, title="What the Model Relies On")
