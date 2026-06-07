from __future__ import annotations

from pathlib import Path

import pandas as pd
import shinyswatch
from shiny import App, reactive, render, ui
from shiny.render import DataGrid
from shinywidgets import output_widget, render_plotly

from src.nlp_dashboard import (
    CLUSTER_OUTPUT_DIR,
    cluster_choice_map,
    cluster_examples_table,
    cluster_metadata_figure,
    cluster_metric_summary,
    cluster_scatter_figure,
    cluster_size_figure,
    cluster_summary_table,
    cluster_timeline_figure,
    filter_cluster_records,
    load_cluster_outputs,
    summarize_visible_clusters,
)
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
APP_SUBTITLE = (
    "A five-chapter story: how documented AI incidents surged, how harm types shifted, "
    "and how public attention concentrates on a few cases."
)


def story_act_banner(act: str, title: str, question: str, takeaway: str):
    return ui.div(
        ui.span(act, class_="story-act-label"),
        ui.h3(title, class_="story-act-title"),
        ui.p(question, class_="story-act-question"),
        ui.p(takeaway, class_="story-act-takeaway"),
        class_="story-act-banner",
    )


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
cluster_outputs = load_cluster_outputs(CLUSTER_OUTPUT_DIR)
year_min = prepared.metadata.get("incident_year_min") or 2010
year_max = prepared.metadata.get("incident_year_max") or 2026
risk_choices = prepared.metadata.get("risk_categories", [])
domain_choices = prepared.metadata.get("source_domains", [])
cluster_choices = cluster_choice_map(cluster_outputs)


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h2("Filters", class_="sidebar-title"),
            ui.p("Filters apply across all story chapters. Incident count = cases; report count = visibility.", class_="sidebar-note"),
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
            class_="hero-card",
        ),
        ui.div(
            ui.span("Story timeline", class_="story-timeline-label"),
            ui.div(
                ui.span("1 · Surge", class_="story-timeline-step"),
                ui.span("→", class_="story-timeline-arrow"),
                ui.span("2 · Shift", class_="story-timeline-step"),
                ui.span("→", class_="story-timeline-arrow"),
                ui.span("3 · Spotlight", class_="story-timeline-step"),
                ui.span("→", class_="story-timeline-arrow"),
                ui.span("4 · Chorus", class_="story-timeline-step"),
                ui.span("→", class_="story-timeline-arrow"),
                ui.span("5 · Connections", class_="story-timeline-step"),
                class_="story-timeline-row",
            ),
            class_="story-timeline",
        ),
        ui.navset_tab(
            ui.nav_panel(
                "1 · The Surge",
                story_act_banner(
                    "Chapter 1",
                    "The record explodes",
                    "How visible are AI incidents over time—and is the growth accelerating?",
                    "Start here: compare incident volume (cases) with report volume (attention).",
                ),
                ui.layout_columns(
                    ui.value_box("Total incidents", ui.output_text("kpi_incidents"), theme="bg-gradient-blue-purple"),
                    ui.value_box("Total reports", ui.output_text("kpi_reports"), theme="bg-gradient-indigo-purple"),
                    ui.value_box("Visible years", ui.output_text("kpi_years"), theme="bg-gradient-green-teal"),
                    ui.value_box("Risk categories", ui.output_text("kpi_risks"), theme="bg-gradient-orange-red"),
                    col_widths=[3, 3, 3, 3],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Incidents vs reports over time"),
                        output_widget("overview_time"),
                        ui.p("Dark line = recorded cases. Teal line = linked media coverage.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Smoothed acceleration"),
                        output_widget("overview_rolling"),
                        ui.p("The 3-year rolling average highlights the long-term surge beyond noisy single years.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[7, 5],
                ),
                ui.card(
                    ui.card_header("Evidence table — filtered incidents"),
                    ui.output_data_frame("incident_table"),
                    ui.p("Sorted by year (newest first). Chapter 3 explores which cases drew the most attention.", class_="chart-note"),
                    class_="soft-card",
                ),
            ),
            ui.nav_panel(
                "2 · The Shift",
                story_act_banner(
                    "Chapter 2",
                    "The villain changes",
                    "Which AI risks dominate—and how has the mix shifted from failures to misuse?",
                    "The plot twist: safety and discrimination led early years; misuse and misinformation rise in recent years.",
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Year × risk category heatmap"),
                        output_widget("risk_heatmap"),
                        ui.p("Scan across years: where does each harm type intensify?", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Composition over time"),
                        output_widget("risk_area"),
                        ui.p("Shares show the changing recipe of harm—not just rising totals.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Risk rank dynamics"),
                        output_widget("risk_bump"),
                        ui.p("When does misuse overtake safety failures in the rankings?", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Category totals (filtered)"),
                        ui.output_data_frame("risk_summary_table"),
                        ui.p("Use the sidebar risk filter to isolate one harm type across all chapters.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[8, 4],
                ),
            ),
            ui.nav_panel(
                "3 · The Spotlight",
                story_act_banner(
                    "Chapter 3",
                    "A few cases swallow the conversation",
                    "Is public attention spread evenly—or concentrated on headline incidents?",
                    "Most incidents have few linked reports; a small elite absorbs most visibility.",
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Top incidents by linked reports"),
                        output_widget("attention_bar"),
                        ui.p("Report count = proxy for media attention, not harm severity.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Case study — incident detail"),
                        ui.output_ui("attention_selector_ui"),
                        ui.output_ui("incident_detail_card"),
                        ui.p("Pick a headline case: title, year, risk label, and source URLs.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[7, 5],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Long-tail distribution"),
                        output_widget("attention_hist"),
                        ui.p("The crowd lives on the left: one or two reports per incident.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Attention concentration (Lorenz curve)"),
                        output_widget("lorenz_curve"),
                        ui.p("Curve above the diagonal = inequality: few incidents, many reports.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "4 · The Chorus",
                story_act_banner(
                    "Chapter 4",
                    "Who tells the story?",
                    "Which news domains appear most often in the linked evidence?",
                    "A concentrated set of publishers shapes what enters the public record.",
                ),
                ui.card(
                    ui.card_header("Top source domains"),
                    output_widget("source_bar"),
                    ui.p(
                        "English-language outlets such as major newspapers and tech press dominate linked reporting. "
                        "Toggle metric in the sidebar to count by incidents or reports.",
                        class_="chart-note",
                    ),
                    class_="soft-card source-card",
                ),
            ),
            ui.nav_panel(
                "5 · Connections",
                story_act_banner(
                    "Chapter 5",
                    "Where it happens—and how it all links",
                    "Geography is incomplete; the network shows how incidents connect to risks and sources.",
                    "Treat the map as directional (~18% location coverage). The network is the connective payoff.",
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Geographic view (limited coverage)"),
                        output_widget("geo_map"),
                        ui.p(
                            "Only a fraction of incidents have reliable location. US-heavy results reflect documentation bias.",
                            class_="chart-note",
                        ),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Top countries / destinations"),
                        output_widget("country_bar"),
                        ui.p("Compare with the map. Few countries have enough cases for strong comparisons.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[8, 4],
                ),
                ui.card(
                    ui.card_header("Incident–risk–source network"),
                    output_widget("network_graph"),
                    ui.p(
                        "Dark = incidents · Teal = risk categories · Gray = source domains. "
                        "Hover to trace how a case links to harm type and media.",
                        class_="chart-note",
                    ),
                    class_="soft-card network-card",
                ),
            ),
            ui.nav_panel(
                "NLP Theme Discovery",
                ui.layout_columns(
                    ui.value_box("Clustered records", ui.output_text("nlp_kpi_records"), theme="bg-gradient-blue-purple"),
                    ui.value_box("Unique incidents", ui.output_text("nlp_kpi_incidents"), theme="bg-gradient-indigo-purple"),
                    ui.value_box("Visible clusters", ui.output_text("nlp_kpi_clusters"), theme="bg-gradient-green-teal"),
                    ui.value_box("Largest theme", ui.output_text("nlp_kpi_largest"), theme="bg-gradient-orange-red"),
                    col_widths=[3, 3, 3, 3],
                ),
                ui.card(
                    ui.card_header("Theme controls"),
                    ui.output_ui("nlp_cluster_selector_ui"),
                    ui.p(
                        "This view uses unsupervised NLP clusters from processed report or incident text.",
                        class_="chart-note",
                    ),
                    class_="soft-card nlp-controls-card",
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("2D text theme map"),
                        output_widget("nlp_cluster_scatter"),
                        ui.p("Each point is a clustered report or incident text record.", class_="chart-note"),
                        class_="soft-card nlp-scatter-card",
                    ),
                    ui.card(
                        ui.card_header("Theme size"),
                        output_widget("nlp_cluster_sizes"),
                        ui.p("Cluster size shows how much of the filtered evidence layer belongs to each discovered theme.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[8, 4],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Theme frequency over time"),
                        output_widget("nlp_cluster_timeline"),
                        ui.p("Use this to compare discovered themes with the incident and report trends.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Metadata profile"),
                        output_widget("nlp_cluster_metadata"),
                        ui.p("This profile connects discovered themes back to risk labels, sectors, failures, and sources when available.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Cluster summary"),
                        ui.output_data_frame("nlp_cluster_summary_table"),
                        ui.p("Top keywords and examples make each discovered theme interpretable.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    ui.card(
                        ui.card_header("Example records"),
                        ui.output_data_frame("nlp_cluster_examples_table"),
                        ui.p("Use examples to validate whether each cluster has a coherent incident theme.", class_="chart-note"),
                        class_="soft-card",
                    ),
                    col_widths=[5, 7],
                ),
            ),
            id="main_tabs",
        ),
        ui.accordion(
            ui.accordion_panel(
                "Data notes & limitations",
                ui.output_ui("data_notes"),
            ),
            open=False,
            class_="data-notes-accordion",
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

    @reactive.calc
    def selected_nlp_cluster():
        try:
            return input.nlp_cluster_filter()
        except Exception:
            return "__all__"

    @reactive.calc
    def nlp_records():
        return filter_cluster_records(
            cluster_outputs.records,
            tuple(input.year_range()),
            list(input.risk_filter()),
            list(input.domain_filter()),
            input.keyword(),
            selected_nlp_cluster(),
        )

    @reactive.calc
    def nlp_summary():
        return summarize_visible_clusters(cluster_outputs.summary, nlp_records())

    @reactive.calc
    def nlp_metrics():
        return cluster_metric_summary(nlp_records(), cluster_outputs.summary)

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
            ui.div(
                ui.span("Category", class_="micro-label"),
                ui.strong(f"{summary['risk_categories']} categories"),
                class_="micro-card micro-card-dark",
            ),
            class_="micro-card-row",
        )

    @output
    @render.ui
    def data_notes():
        items = [
            "This database contains reported or documented AI incidents, not every real-world incident.",
            "Report count reflects public attention and media visibility, not necessarily severity.",
            "Geographic views depend on available location metadata (~18% of incidents have location).",
            "Classification coverage varies across MIT, GMF, and CSET taxonomies.",
        ] + prepared.notes[:6]
        return ui.div(
            ui.tags.ul(*[ui.tags.li(item) for item in items]),
            class_="note-card note-card-inline",
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
        frame = attention().sort_values(["incident_year", "incident_title"], ascending=[False, True]).copy()
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
        return DataGrid(frame, summary="Filtered incidents (newest year first, up to 50 rows)")

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

    @output
    @render.text
    def nlp_kpi_records():
        return f"{nlp_metrics()['records']:,}"

    @output
    @render.text
    def nlp_kpi_incidents():
        return f"{nlp_metrics()['incidents']:,}"

    @output
    @render.text
    def nlp_kpi_clusters():
        return f"{nlp_metrics()['clusters']:,}"

    @output
    @render.text
    def nlp_kpi_largest():
        return str(nlp_metrics()["largest_cluster"])

    @output
    @render.ui
    def nlp_cluster_selector_ui():
        if cluster_outputs.records.empty:
            command = (
                "python cluster_incidents.py --data-path processed_data/report_level_processed.csv "
                "--output-dir cluster_outputs/sbert_report_clusters"
            )
            items = cluster_outputs.notes or ["Cluster outputs are unavailable."]
            return ui.div(
                ui.p("No NLP cluster output has been loaded yet.", class_="muted-text"),
                ui.tags.ul(*[ui.tags.li(item) for item in items[:4]]),
                ui.p("Run this command, then restart the dashboard:", class_="muted-text"),
                ui.tags.code(command),
                class_="note-card note-card-inline",
            )
        return ui.input_select(
            "nlp_cluster_filter",
            "Theme cluster",
            choices=cluster_choices,
            selected="__all__",
        )

    @output
    @render_plotly
    def nlp_cluster_scatter():
        return cluster_scatter_figure(nlp_records())

    @output
    @render_plotly
    def nlp_cluster_sizes():
        return cluster_size_figure(cluster_outputs.summary, nlp_records())

    @output
    @render_plotly
    def nlp_cluster_timeline():
        return cluster_timeline_figure(nlp_records())

    @output
    @render_plotly
    def nlp_cluster_metadata():
        return cluster_metadata_figure(nlp_records())

    @output
    @render.data_frame
    def nlp_cluster_summary_table():
        frame = cluster_summary_table(cluster_outputs.summary, nlp_records())
        return DataGrid(frame, summary="NLP cluster summary")

    @output
    @render.data_frame
    def nlp_cluster_examples_table():
        frame = cluster_examples_table(nlp_records())
        return DataGrid(frame, summary="Example clustered records")


app = App(app_ui, server, static_assets=Path(__file__).parent / "assets")
