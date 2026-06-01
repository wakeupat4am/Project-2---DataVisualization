from __future__ import annotations

from pathlib import Path

import pandas as pd
import shinyswatch
from shiny import App, reactive, render, ui
from shiny.render import DataGrid
from shinywidgets import output_widget, render_plotly

from src.preprocess import PreparedData, build_filtered_context, prepare_data
from src.visualizations import (
    bump_chart_figure,
    empty_figure,
    heatmap_figure,
    incident_attention_bar,
    line_overview_figure,
    long_tail_histogram,
    lorenz_curve_figure,
    map_figure,
    network_figure,
    rolling_average_figure,
    source_domain_bar,
    stacked_area_figure,
    country_bar as country_bar_figure,
)

try:
    import pycountry
except ImportError:  # pragma: no cover - optional runtime dependency
    pycountry = None


APP_TITLE = "AI Incident Database Explorer"
APP_SUBTITLE = "Temporal, geographic, and network visual analytics of AI incidents and public attention."


def iso2_to_iso3(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"global", "--", "unknown"}:
        return None
    if pycountry is None:
        return text if len(text) == 3 and text.isalpha() else None
    if len(text) == 2 and text.isalpha():
        country = pycountry.countries.get(alpha_2=text.upper())
        return getattr(country, "alpha_3", None)
    country = pycountry.countries.get(name=text)
    return getattr(country, "alpha_3", None)


def make_metric_summary(context: dict[str, pd.DataFrame], prepared: PreparedData) -> dict[str, int | str]:
    incidents = context["incidents"]
    incident_reports = context["incident_reports"]
    years = incidents["incident_year"].dropna()
    return {
        "total_incidents": int(incidents["incident_id"].nunique()) if not incidents.empty else 0,
        "total_reports": int(incident_reports["report_id"].nunique()) if not incident_reports.empty else 0,
        "year_range": f"{int(years.min())} - {int(years.max())}" if not years.empty else "N/A",
        "risk_categories": len(prepared.metadata.get("risk_categories", [])),
    }


def aggregate_year_counts(context: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    incidents = context["incidents"]
    incident_reports = context["incident_reports"]
    incident_year = (
        incidents.dropna(subset=["incident_year"])
        .groupby("incident_year", as_index=False)
        .size()
        .rename(columns={"incident_year": "year", "size": "count"})
    )
    report_year = (
        incident_reports.dropna(subset=["report_year"])
        .groupby("report_year", as_index=False)
        .size()
        .rename(columns={"report_year": "year", "size": "count"})
    )
    return incident_year, report_year


def aggregate_risk_by_year(context: dict[str, pd.DataFrame], metric: str, top_n: int) -> pd.DataFrame:
    risk_long = context["risk_long"]
    incidents = context["incidents"]
    incident_reports = context["incident_reports"]
    if risk_long.empty:
        return pd.DataFrame(columns=["year", "risk_category", "count", "share"])

    if metric == "reports" and not incident_reports.empty:
        base = incident_reports.merge(risk_long, on="incident_id", how="inner")
        year_col = "report_year" if "report_year" in base.columns else "incident_year"
        frame = (
            base.dropna(subset=[year_col, "risk_category"])
            .groupby([year_col, "risk_category"], as_index=False)
            .size()
            .rename(columns={year_col: "year", "size": "count"})
        )
    else:
        base = incidents[["incident_id", "incident_year"]].merge(risk_long, on="incident_id", how="inner")
        frame = (
            base.dropna(subset=["incident_year", "risk_category"])
            .groupby(["incident_year", "risk_category"], as_index=False)
            .size()
            .rename(columns={"incident_year": "year", "size": "count"})
        )

    if frame.empty:
        return frame

    keep = (
        frame.groupby("risk_category", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(top_n)["risk_category"]
        .tolist()
    )
    frame = frame[frame["risk_category"].isin(keep)]
    totals = frame.groupby("year")["count"].transform("sum").replace(0, pd.NA)
    frame["share"] = frame["count"] / totals
    return frame.sort_values(["year", "count"], ascending=[True, False])


def attention_table(context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    incidents = context["incidents"]
    incident_reports = context["incident_reports"]
    if incidents.empty:
        return pd.DataFrame(columns=["incident_id", "incident_title", "incident_year", "report_count"])
    if incident_reports.empty:
        return incidents[["incident_id", "incident_title", "incident_year", "report_count"]].copy()
    counts = incident_reports.groupby("incident_id", as_index=False)["report_id"].nunique().rename(
        columns={"report_id": "report_count"}
    )
    frame = incidents[["incident_id", "incident_title", "incident_year", "incident_description"]].merge(
        counts, on="incident_id", how="left"
    )
    frame["report_count"] = frame["report_count"].fillna(0).astype(int)
    return frame


def top_domains_table(context: dict[str, pd.DataFrame], top_n: int) -> pd.DataFrame:
    incident_reports = context["incident_reports"]
    if incident_reports.empty or "source_domain" not in incident_reports.columns:
        return pd.DataFrame(columns=["source_domain", "count"])
    return (
        incident_reports.groupby("source_domain", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def country_counts_table(context: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    locations = context["locations"]
    incident_reports = context["incident_reports"]
    if locations.empty:
        return pd.DataFrame(columns=["country", "country_label", "count", "iso_alpha"])

    if metric == "reports" and not incident_reports.empty:
        frame = locations.merge(incident_reports[["incident_id", "report_id"]], on="incident_id", how="inner")
        grouped = frame.groupby(["country", "location_label"], as_index=False)["report_id"].nunique().rename(columns={"report_id": "count"})
    else:
        grouped = locations.groupby(["country", "location_label"], as_index=False)["incident_id"].nunique().rename(columns={"incident_id": "count"})

    grouped["country"] = grouped["country"].replace("", pd.NA)
    grouped["location_label"] = grouped["location_label"].replace("", pd.NA)
    grouped["country_label"] = grouped["country"].fillna(grouped["location_label"]).astype(str).str.strip()
    grouped = grouped[grouped["country_label"].ne("")]
    grouped = (
        grouped.groupby(["country_label"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
    grouped["country"] = grouped["country_label"]
    grouped["iso_alpha"] = grouped["country"].apply(iso2_to_iso3)
    return grouped.sort_values("count", ascending=False)


def network_table(context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    incidents = context["incidents"]
    risk_long = context["risk_long"]
    incident_reports = context["incident_reports"]
    if incidents.empty or risk_long.empty or incident_reports.empty:
        return pd.DataFrame(columns=["incident_id", "incident_title", "risk_category", "source_domain", "report_count"])

    attention = attention_table(context)[["incident_id", "incident_title", "report_count"]]
    domains = (
        incident_reports.groupby(["incident_id", "source_domain"], as_index=False)["report_id"]
        .nunique()
        .rename(columns={"report_id": "domain_reports"})
    )
    frame = attention.merge(risk_long[["incident_id", "risk_category"]].drop_duplicates(), on="incident_id", how="inner")
    frame = frame.merge(domains, on="incident_id", how="left")
    return frame


prepared = prepare_data("data")
year_min = prepared.metadata.get("incident_year_min") or 2010
year_max = prepared.metadata.get("incident_year_max") or 2026
risk_choices = prepared.metadata.get("risk_categories", [])
domain_choices = prepared.metadata.get("source_domains", [])


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h2("Filters", class_="sidebar-title"),
            ui.p("Use these controls to compare incidents with public attention.", class_="sidebar-note"),
            class_="sidebar-header",
        ),
        ui.div("Time & scope", class_="sidebar-section-label"),
        ui.input_slider(
            "year_range",
            "Year range",
            min=year_min,
            max=year_max,
            value=(year_min, year_max),
            step=1,
        ),
        ui.div("Narrative filters", class_="sidebar-section-label"),
        ui.input_selectize(
            "risk_filter",
            "Risk/category selector",
            choices=risk_choices,
            selected=[],
            multiple=True,
            options={"placeholder": "All risk categories"},
        ),
        ui.input_selectize(
            "domain_filter",
            "Source/domain selector",
            choices=domain_choices,
            selected=[],
            multiple=True,
            options={"placeholder": "All source domains"},
        ),
        ui.input_text("keyword", "Keyword search", placeholder="Search title or description"),
        ui.div("View controls", class_="sidebar-section-label"),
        ui.input_radio_buttons(
            "metric_mode",
            "Metric",
            choices={"incidents": "Count by incidents", "reports": "Count by reports"},
            selected="incidents",
        ),
        ui.input_numeric("top_n", "Top N for ranking charts", value=10, min=5, max=25, step=1),
        ui.input_select(
            "map_projection",
            "Map projection",
            choices={
                "natural earth": "Natural earth",
                "orthographic": "Orthographic globe",
                "equirectangular": "Equirectangular",
            },
            selected="orthographic",
        ),
        ui.div(
            ui.h5("Reading tip"),
            ui.p(
                "Incident count measures recorded cases. Report count measures visibility and public attention.",
                class_="sidebar-note",
            ),
            class_="sidebar-tip",
        ),
        width=340,
        class_="app-sidebar",
    ),
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="styles.css"),
    ),
    ui.div(
        ui.div(
            ui.h1(APP_TITLE, class_="app-title"),
            ui.p(APP_SUBTITLE, class_="app-subtitle"),
            ui.output_ui("hero_microstats"),
            ui.div(
                ui.span("Story flow", class_="story-chip"),
                ui.span("Overview", class_="story-chip story-chip-active"),
                ui.span("Risk Evolution", class_="story-chip"),
                ui.span("Attention", class_="story-chip"),
                ui.span("Geography + Network", class_="story-chip"),
                class_="story-chip-row",
            ),
            class_="hero-card",
        ),
        ui.output_ui("data_notes"),
        ui.navset_tab(
            ui.nav_panel(
                "Overview",
                ui.layout_columns(
                    ui.value_box("Total incidents", ui.output_text("kpi_incidents"), theme="bg-gradient-blue-purple"),
                    ui.value_box("Total reports", ui.output_text("kpi_reports"), theme="bg-gradient-indigo-purple"),
                    ui.value_box("Visible years", ui.output_text("kpi_years"), theme="bg-gradient-green-teal"),
                    ui.value_box("Risk categories", ui.output_text("kpi_risks"), theme="bg-gradient-orange-red"),
                    col_widths=[3, 3, 3, 3],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("How visible are AI incidents over time?"),
                        output_widget("overview_time"),
                        ui.p("Incidents and reports move together, but they are not the same metric.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Smoothed trend"),
                        output_widget("overview_rolling"),
                        ui.p("The rolling average makes longer-term acceleration easier to read.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[7, 5],
                ),
                ui.card(
                    ui.card_header("Filtered incidents"),
                    ui.output_data_frame("incident_table"),
                    ui.p("This table updates from the shared filters and keeps the story grounded in examples.", class_="chart-note"),
                    class_="soft-card",
                ),
            ),
            ui.nav_panel(
                "Risk Evolution",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Year × Risk category heatmap"),
                        output_widget("risk_heatmap"),
                        ui.p("This view answers which risks become more visible over time.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Composition over time"),
                        output_widget("risk_area"),
                        ui.p("Area shares show how the mix of AI risks changes rather than only the volume.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Risk rank dynamics"),
                        output_widget("risk_bump"),
                        ui.p("The bump chart shows when categories move up or down in prominence.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Filtered category summary"),
                        ui.output_data_frame("risk_summary_table"),
                        ui.p("The current risk filter doubles as the main interaction for narrowing the incident list.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[8, 4],
                ),
            ),
            ui.nav_panel(
                "Attention & Sources",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Top incidents by public attention"),
                        output_widget("attention_bar"),
                        ui.p("Linked report count acts as a proxy for visibility and media attention.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Attention detail"),
                        ui.output_ui("attention_selector_ui"),
                        ui.output_ui("incident_detail_card"),
                        ui.p("Choose an incident to inspect its title, timing, description, and linked reports.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[7, 5],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Long-tail distribution"),
                        output_widget("attention_hist"),
                        ui.p("Most incidents receive limited attention while a few dominate the conversation.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Source domains"),
                        output_widget("source_bar"),
                        ui.p("This shows which publishers appear most often in the evidence layer.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Attention concentration"),
                        output_widget("lorenz_curve"),
                        ui.p("The Lorenz curve highlights whether reporting is evenly distributed or highly concentrated.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[4, 4, 4],
                ),
            ),
            ui.nav_panel(
                "Geographic & Advanced View",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Where do AI incidents appear?"),
                        output_widget("geo_map"),
                        ui.p(
                            "Geographic results depend on available location metadata. Missing or ambiguous locations are excluded.",
                            class_="chart-note",
                        ),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Top countries / destinations"),
                        output_widget("country_bar"),
                        ui.p("Use this together with the map to compare geographic concentration.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[8, 4],
                ),
                ui.card(
                    ui.card_header("Incident–Risk–Source network"),
                    output_widget("network_graph"),
                    ui.p("This advanced view connects incidents to both risk categories and reporting domains.", class_="chart-note"),
                    class_="soft-card",
                ),
            ),
            id="main_tabs",
        ),
        class_="app-shell",
    ),
    title="AI Incident Database Explorer",
    fillable=True,
    theme=shinyswatch.theme.flatly(),
)


def server(input, output, session):
    @reactive.calc
    def filtered_context():
        return build_filtered_context(
            prepared,
            tuple(input.year_range()),
            list(input.risk_filter()),
            list(input.domain_filter()),
            input.keyword(),
        )

    @reactive.calc
    def metrics():
        return make_metric_summary(filtered_context(), prepared)

    @reactive.calc
    def incident_years():
        return aggregate_year_counts(filtered_context())

    @reactive.calc
    def risk_years():
        return aggregate_risk_by_year(filtered_context(), input.metric_mode(), int(input.top_n()))

    @reactive.calc
    def attention():
        return attention_table(filtered_context())

    @reactive.calc
    def domains():
        return top_domains_table(filtered_context(), int(input.top_n()))

    @reactive.calc
    def countries():
        return country_counts_table(filtered_context(), input.metric_mode())

    @reactive.calc
    def network_data():
        return network_table(filtered_context())

    @output
    @render.text
    def kpi_incidents():
        return f"{metrics()['total_incidents']:,}"

    @output
    @render.text
    def kpi_reports():
        return f"{metrics()['total_reports']:,}"

    @output
    @render.text
    def kpi_years():
        return metrics()["year_range"]

    @output
    @render.text
    def kpi_risks():
        return f"{metrics()['risk_categories']:,}"

    @output
    @render.ui
    def hero_microstats():
        summary = metrics()
        return ui.div(
            ui.div(ui.span("Incidents", class_="micro-label"), ui.strong(f"{summary['total_incidents']:,}"), class_="micro-card"),
            ui.div(ui.span("Reports", class_="micro-label"), ui.strong(f"{summary['total_reports']:,}"), class_="micro-card"),
            ui.div(ui.span("Years", class_="micro-label"), ui.strong(summary["year_range"]), class_="micro-card"),
            ui.div(ui.span("Taxonomy", class_="micro-label"), ui.strong(f"{summary['risk_categories']} risk types"), class_="micro-card micro-card-dark"),
            class_="micro-card-row",
        )

    @output
    @render.ui
    def data_notes():
        if not prepared.notes:
            return ui.div(
                ui.h4("Data Notes & Limitations"),
                ui.tags.ul(
                    ui.tags.li("This database contains reported or documented AI incidents, not every real-world incident."),
                    ui.tags.li("Report count reflects public attention and media visibility, not necessarily severity."),
                    ui.tags.li("Geographic views depend on available location metadata."),
                    ui.tags.li("Classification coverage varies across MIT, GMF, and CSET taxonomies."),
                ),
                class_="note-card",
            )

        items = [
            "This database contains reported or documented AI incidents, not every real-world incident.",
            "Report count reflects public attention and media visibility, not necessarily severity.",
            "Geographic views depend on available location metadata.",
            "Classification coverage varies across MIT, GMF, and CSET taxonomies.",
        ] + prepared.notes[:6]
        return ui.div(
            ui.h4("Data Notes & Limitations"),
            ui.tags.ul(*[ui.tags.li(item) for item in items]),
            class_="note-card",
        )

    @output
    @render_plotly
    def overview_time():
        incidents_by_year, reports_by_year = incident_years()
        return line_overview_figure(incidents_by_year, reports_by_year)

    @output
    @render_plotly
    def overview_rolling():
        incidents_by_year, reports_by_year = incident_years()
        if input.metric_mode() == "reports":
            return rolling_average_figure(reports_by_year, "Reports")
        return rolling_average_figure(incidents_by_year, "Incidents")

    @output
    @render.data_frame
    def incident_table():
        frame = attention().sort_values(["report_count", "incident_year"], ascending=[False, False]).copy()
        if frame.empty:
            frame = pd.DataFrame({"message": ["No incidents match the current filters."]})
        else:
            frame = frame.rename(
                columns={
                    "incident_title": "Title",
                    "incident_year": "Year",
                    "incident_description": "Description",
                    "report_count": "Report count",
                }
            )[["Title", "Year", "Report count", "Description"]]
            frame = frame.head(50)
        return DataGrid(frame, summary="Top 50 filtered incidents")

    @output
    @render_plotly
    def risk_heatmap():
        metric_label = "Reports" if input.metric_mode() == "reports" else "Incidents"
        return heatmap_figure(risk_years(), metric_label)

    @output
    @render_plotly
    def risk_area():
        metric_label = "Report count" if input.metric_mode() == "reports" else "Incident count"
        return stacked_area_figure(risk_years(), metric_label)

    @output
    @render_plotly
    def risk_bump():
        return bump_chart_figure(risk_years())

    @output
    @render.data_frame
    def risk_summary_table():
        frame = risk_years()
        if frame.empty:
            frame = pd.DataFrame({"message": ["No risk categories are available for the current filters."]})
        else:
            frame = (
                frame.groupby("risk_category", as_index=False)["count"]
                .sum()
                .sort_values("count", ascending=False)
                .rename(columns={"risk_category": "Risk category", "count": "Visible count"})
            )
        return DataGrid(frame, summary="Risk category totals")

    @output
    @render_plotly
    def attention_bar():
        frame = attention().sort_values("report_count", ascending=False).head(int(input.top_n()))
        return incident_attention_bar(frame)

    @output
    @render_plotly
    def attention_hist():
        return long_tail_histogram(attention())

    @output
    @render_plotly
    def source_bar():
        return source_domain_bar(domains())

    @output
    @render_plotly
    def lorenz_curve():
        return lorenz_curve_figure(attention())

    @output
    @render.ui
    def attention_selector_ui():
        frame = attention().sort_values("report_count", ascending=False).head(50)
        choices = {row["incident_id"]: row["incident_title"][:100] for _, row in frame.iterrows()}
        if not choices:
            return ui.p("No incidents are available for detail view.", class_="muted-text")
        selected = next(iter(choices.keys()))
        return ui.input_selectize("selected_incident", "Select an incident", choices=choices, selected=selected)

    @output
    @render.ui
    def incident_detail_card():
        try:
            selected = input.selected_incident()
        except Exception:
            selected = None
        frame = attention()
        if frame.empty:
            return ui.p("Incident detail is unavailable.", class_="muted-text")
        if not selected or selected not in frame["incident_id"].astype(str).tolist():
            selected = frame.sort_values("report_count", ascending=False)["incident_id"].iloc[0]

        row = frame[frame["incident_id"].astype(str) == str(selected)].iloc[0]
        detail_risks = (
            filtered_context()["risk_long"]
            .loc[lambda df: df["incident_id"].astype(str) == str(selected), "risk_category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        detail_reports = (
            filtered_context()["incident_reports"]
            .loc[lambda df: df["incident_id"].astype(str) == str(selected), ["report_title", "report_url", "source_domain"]]
            .drop_duplicates()
            .head(5)
        )
        report_list = (
            ui.tags.ul(
                *[
                    ui.tags.li(
                        ui.a(
                            report["report_title"] or report["source_domain"],
                            href=report["report_url"],
                            target="_blank",
                        )
                        if str(report.get("report_url", "")).startswith("http")
                        else f"{report['report_title'] or report['source_domain']}"
                    )
                    for _, report in detail_reports.iterrows()
                ]
            )
            if not detail_reports.empty
            else ui.p("No linked source URLs are available.", class_="muted-text")
        )
        return ui.div(
            ui.h4(row["incident_title"] or "Untitled incident"),
            ui.p(f"Year: {row['incident_year'] if pd.notna(row['incident_year']) else 'N/A'}"),
            ui.p(f"Linked reports: {int(row['report_count'])}"),
            ui.p(f"Risk categories: {', '.join(detail_risks[:5]) if detail_risks else 'Not available'}"),
            ui.p(row["incident_description"] or "No description available."),
            ui.h5("Source examples"),
            report_list,
            class_="detail-card",
        )

    @output
    @render_plotly
    def geo_map():
        metric_label = "Reports" if input.metric_mode() == "reports" else "Incidents"
        return map_figure(countries(), input.map_projection(), metric_label)

    @output
    @render_plotly
    def country_bar():
        return country_bar_figure(countries())

    @output
    @render_plotly
    def network_graph():
        return network_figure(network_data(), max_incidents=max(10, int(input.top_n())))


app = App(app_ui, server, static_assets=Path(__file__).parent / "assets")
