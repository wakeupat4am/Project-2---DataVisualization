from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = [
    "#79c7c5",
    "#2b2d42",
    "#95d5b2",
    "#f4a261",
    "#cdb4db",
    "#4a5568",
    "#84a59d",
    "#ffb4a2",
]


RISK_ABBREVIATIONS = {
    "1. Discrimination and Toxicity": "D&T",
    "2. Privacy & Security": "Privacy",
    "3. Misinformation": "Misinfo",
    "4. Malicious Actors & Misuse": "Misuse",
    "5. Human-Computer Interaction": "HCI",
    "6. Socioeconomic & Environmental Harms": "Socio-env",
    "7. AI system safety, failures, and limitations": "Safety",
}


def short_label(value: str, max_len: int = 22) -> str:
    text = str(value)
    if text in RISK_ABBREVIATIONS:
        return RISK_ABBREVIATIONS[text]
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#4a5568"},
    )
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def line_overview_figure(incidents_by_year: pd.DataFrame, reports_by_year: pd.DataFrame) -> go.Figure:
    if incidents_by_year.empty and reports_by_year.empty:
        return empty_figure("No time-series data is available for the current filters.")

    fig = go.Figure()
    if not incidents_by_year.empty:
        fig.add_trace(
            go.Scatter(
                x=incidents_by_year["year"],
                y=incidents_by_year["count"],
                mode="lines+markers",
                name="Incident count",
                line={"color": PALETTE[1], "width": 3},
                marker={"size": 8},
                hovertemplate="Year %{x}<br>Incidents %{y}<extra></extra>",
            )
        )
    if not reports_by_year.empty:
        fig.add_trace(
            go.Scatter(
                x=reports_by_year["year"],
                y=reports_by_year["count"],
                mode="lines+markers",
                name="Report count",
                line={"color": PALETTE[0], "width": 3},
                marker={"size": 8},
                hovertemplate="Year %{x}<br>Reports %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=90),
        legend=dict(orientation="h", y=-0.24, x=0, title=None),
        xaxis_title="Year",
        yaxis_title="Count",
    )
    return fig


def rolling_average_figure(year_counts: pd.DataFrame, label: str) -> go.Figure:
    if year_counts.empty:
        return empty_figure("Rolling trend is unavailable because yearly counts are missing.")
    frame = year_counts.sort_values("year").copy()
    frame["rolling"] = frame["count"].rolling(window=3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["year"],
            y=frame["count"],
            name=label,
            marker_color="#dceef0",
            hovertemplate="Year %{x}<br>Count %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["year"],
            y=frame["rolling"],
            mode="lines+markers",
            name="3-year rolling average",
            line={"color": PALETTE[0], "width": 3},
            hovertemplate="Year %{x}<br>Rolling avg %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=90),
        xaxis_title="Year",
        yaxis_title="Count",
        legend=dict(
            orientation="h",
            y=-0.24,
            x=0,
            title=None,
        ),
    )
    return fig


def heatmap_figure(risk_by_year: pd.DataFrame, metric_label: str) -> go.Figure:
    if risk_by_year.empty:
        return empty_figure("Risk categories are unavailable for the current filters.")
    pivot = risk_by_year.pivot(index="risk_category", columns="year", values="count").fillna(0)
    short_index = [short_label(item, 18) for item in pivot.index]
    share = pivot.div(pivot.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    hover_text = [
        [
            f"Category: {category}<br>Year: {year}<br>{metric_label}: {int(pivot.loc[category, year])}<br>Share: {share.loc[category, year]:.1%}"
            for year in pivot.columns
        ]
        for category in pivot.index
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=short_index,
            colorscale=[[0, "#eef6f6"], [0.5, "#79c7c5"], [1, "#2b2d42"]],
            text=hover_text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Year",
        yaxis_title="Risk category",
    )
    return fig


def stacked_area_figure(risk_by_year: pd.DataFrame, metric_label: str) -> go.Figure:
    if risk_by_year.empty:
        return empty_figure("Risk composition cannot be drawn without category labels.")
    frame = risk_by_year.copy()
    frame["risk_short"] = frame["risk_category"].apply(short_label)
    fig = px.area(
        frame,
        x="year",
        y="count",
        color="risk_short",
        color_discrete_sequence=PALETTE,
        custom_data=["share", "risk_category"],
    )
    fig.update_traces(
        hovertemplate="Year %{x}<br>%{customdata[1]}<br>"
        + f"{metric_label}: "
        + "%{y}<br>Share %{customdata[0]:.1%}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=120),
        xaxis_title="Year",
        yaxis_title=metric_label,
        legend_title=None,
        legend=dict(orientation="h", y=-0.34, x=0),
    )
    return fig


def bump_chart_figure(risk_by_year: pd.DataFrame) -> go.Figure:
    if risk_by_year.empty:
        return empty_figure("Category rank trends are unavailable.")
    rank_frame = risk_by_year.copy()
    rank_frame["risk_short"] = rank_frame["risk_category"].apply(short_label)
    rank_frame["rank"] = rank_frame.groupby("year")["count"].rank(method="dense", ascending=False)
    fig = px.line(
        rank_frame,
        x="year",
        y="rank",
        color="risk_short",
        markers=True,
        color_discrete_sequence=PALETTE,
        custom_data=["risk_category"],
    )
    fig.update_yaxes(autorange="reversed", dtick=1, title="Rank (1 = most visible)")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=120),
        xaxis_title="Year",
        legend_title=None,
        legend=dict(orientation="h", y=-0.34, x=0),
    )
    fig.update_traces(hovertemplate="Year %{x}<br>%{customdata[0]}<br>Rank %{y}<extra></extra>")
    return fig


def incident_attention_bar(top_incidents: pd.DataFrame) -> go.Figure:
    if top_incidents.empty:
        return empty_figure("No incident attention data is available for the current filters.")
    frame = top_incidents.sort_values("report_count")
    frame["title_short"] = frame["incident_title"].apply(lambda value: short_label(value, 38))
    fig = px.bar(
        frame,
        x="report_count",
        y="title_short",
        orientation="h",
        color="report_count",
        color_continuous_scale=["#dceef0", "#79c7c5", "#2b2d42"],
    )
    fig.update_traces(
        hovertemplate="Incident %{customdata[2]}<br>Reports %{x}<extra></extra>",
        customdata=np.stack([frame["incident_id"], frame["incident_year"], frame["incident_title"]], axis=-1),
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_showscale=False,
        xaxis_title="Linked reports",
        yaxis_title="Incident",
    )
    return fig


def long_tail_histogram(attention: pd.DataFrame) -> go.Figure:
    if attention.empty:
        return empty_figure("The report-per-incident distribution is unavailable.")
    fig = px.histogram(attention, x="report_count", nbins=min(25, max(5, attention["report_count"].nunique())))
    fig.update_traces(marker_color=PALETTE[0], hovertemplate="Reports %{x}<br>Incidents %{y}<extra></extra>")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Reports per incident",
        yaxis_title="Number of incidents",
    )
    return fig


def source_domain_bar(domains: pd.DataFrame) -> go.Figure:
    if domains.empty:
        return empty_figure("No source domains are available for the current filters.")
    fig = px.bar(domains.sort_values("count"), x="count", y="source_domain", orientation="h", color="count")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_showscale=False,
        xaxis_title="Reports",
        yaxis_title="Source domain",
    )
    fig.update_traces(hovertemplate="Domain %{y}<br>Reports %{x}<extra></extra>")
    return fig


def lorenz_curve_figure(attention: pd.DataFrame) -> go.Figure:
    if attention.empty or attention["report_count"].sum() == 0:
        return empty_figure("Lorenz curve requires report counts linked to incidents.")
    sorted_counts = np.sort(attention["report_count"].to_numpy())
    cumulative_incidents = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    cumulative_reports = np.cumsum(sorted_counts) / sorted_counts.sum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.insert(cumulative_incidents, 0, 0),
            y=np.insert(cumulative_reports, 0, 0),
            mode="lines",
            name="Observed attention",
            line={"color": PALETTE[1], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Even attention",
            line={"dash": "dash", "color": "#a0aec0"},
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=80),
        xaxis_title="Cumulative share of incidents",
        yaxis_title="Cumulative share of reports",
        legend=dict(orientation="h", y=-0.24, x=0, title=None),
    )
    return fig


def map_figure(country_counts: pd.DataFrame, projection: str, metric_label: str) -> go.Figure:
    if country_counts.empty:
        return empty_figure("Country metadata is missing, so the geographic view cannot be drawn.")

    has_iso = "iso_alpha" in country_counts.columns and country_counts["iso_alpha"].notna().any()
    location_col = "iso_alpha" if has_iso else "country_label"
    frame = country_counts.copy()
    if has_iso:
        frame = frame[frame["iso_alpha"].notna()]

    fig = px.choropleth(
        frame,
        locations=location_col,
        locationmode="ISO-3" if has_iso else "country names",
        color="count",
        hover_name="country_label",
        color_continuous_scale=["#eef6f6", "#79c7c5", "#2b2d42"],
        projection=projection,
    )
    fig.update_traces(hovertemplate="Country %{hovertext}<br>" + f"{metric_label}: " + "%{z}<extra></extra>")
    if has_iso:
        bubble_frame = frame.dropna(subset=["iso_alpha"]).copy()
        if not bubble_frame.empty:
            fig.add_trace(
                go.Scattergeo(
                    locations=bubble_frame["iso_alpha"],
                    locationmode="ISO-3",
                    text=[
                        f"{row.country_label}<br>{metric_label}: {int(row.count)}"
                        for row in bubble_frame.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    mode="markers",
                    marker=dict(
                        size=np.clip(np.sqrt(bubble_frame["count"]) * 5, 8, 28),
                        color="#101828",
                        opacity=0.7,
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    name="Incident intensity",
                    showlegend=False,
                )
            )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#ffffff",
        showcountries=True,
        countrycolor="#d7e5e5",
        showland=True,
        landcolor="#f7fbfb",
        showocean=True,
        oceancolor="#dceef0",
        showlakes=True,
        lakecolor="#dceef0",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar_title=metric_label,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def country_bar(country_counts: pd.DataFrame) -> go.Figure:
    if country_counts.empty:
        return empty_figure("No country counts are available.")
    frame = country_counts.sort_values("count").tail(12)
    fig = px.bar(frame, x="count", y="country_label", orientation="h", color="count")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False,
        xaxis_title="Count",
        yaxis_title="Country",
    )
    return fig


def network_figure(network_frame: pd.DataFrame, max_incidents: int = 25) -> go.Figure:
    if network_frame.empty:
        return empty_figure("Network view is unavailable because linked incidents, risks, or sources are missing.")

    top_incidents = (
        network_frame.groupby(["incident_id", "incident_title"], as_index=False)["report_count"]
        .max()
        .sort_values("report_count", ascending=False)
        .head(max_incidents)
    )
    frame = network_frame[network_frame["incident_id"].isin(top_incidents["incident_id"])]
    if frame.empty:
        return empty_figure("Current filters leave too few nodes for the network view.")

    graph = nx.Graph()
    for _, row in frame.iterrows():
        incident_node = f"incident::{row['incident_id']}"
        graph.add_node(
            incident_node,
            label=row["incident_title"][:80] or row["incident_id"],
            node_type="Incident",
            color=PALETTE[1],
            size=18,
        )
        if row.get("risk_category"):
            risk_node = f"risk::{row['risk_category']}"
            graph.add_node(risk_node, label=row["risk_category"], node_type="Risk", color=PALETTE[0], size=24)
            graph.add_edge(incident_node, risk_node)
        if row.get("source_domain"):
            source_node = f"source::{row['source_domain']}"
            graph.add_node(
                source_node,
                label=row["source_domain"],
                node_type="Source",
                color=PALETTE[4],
                size=20,
            )
            graph.add_edge(incident_node, source_node)

    if len(graph.nodes) == 0:
        return empty_figure("The current network selection contains no nodes.")

    pos = nx.spring_layout(graph, seed=7, k=1.2 / math.sqrt(max(len(graph.nodes), 1)))
    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=1, color="#d9e6e8"),
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_type = []
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{attrs['node_type']}: {attrs['label']}")
        node_color.append(attrs["color"])
        node_size.append(attrs["size"])
        node_type.append(attrs["node_type"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(color=node_color, size=node_size, line=dict(color="white", width=1.5)),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="Incident nodes connect to risk categories and source domains.",
                xref="paper",
                yref="paper",
                x=0.01,
                y=1.05,
                showarrow=False,
                font=dict(size=12, color="#5f6c7b"),
            )
        ],
    )
    return fig
