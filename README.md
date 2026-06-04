# AI Incident Database Explorer

Interactive Python Shiny dashboard for the AI Incident Database.

**Presentation story:**  
*From AI Incidents to Public Attention: How AI Risks Evolve and Spread Over Time.*

The app is designed for a 6-minute demo that moves from overall visibility, to risk evolution, to attention concentration, to geography and network context.

## Dashboard Story
The dashboard answers four connected questions:

1. **How visible are AI incidents over time?**
2. **Which AI risks are becoming more visible?**
3. **Which incidents and source domains dominate public attention?**
4. **Where do AI incidents appear, and how are incidents connected to risks and sources?**

It explicitly separates:
- **Incident count**: how many AI incidents are recorded
- **Report count**: how much reporting/public attention sits behind those incidents

## Project Structure
- [app.py](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/app.py): main Python Shiny app
- [requirements.txt](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/requirements.txt): Python dependencies
- [assets/styles.css](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/assets/styles.css): custom light dashboard styling
- [src/data_loader.py](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/src/data_loader.py): CSV loading utilities
- [src/preprocess.py](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/src/preprocess.py): robust preprocessing, auto-detection, joins, and aggregates
- [src/visualizations.py](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/src/visualizations.py): reusable Plotly chart builders
- [data](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data): expected data folder

## Expected Data Files
The app looks for CSV files in `data/` and is designed to tolerate missing optional files.

Primary files:
- `incidents.csv`: one row per AI incident
- `reports.csv`: one row per linked report/article
- `classifications_*.csv`: taxonomy files such as MIT, GMF, CSET

Optional files:
- `duplicates.csv`
- `submissions.csv`
- `quickadd.csv`

If some files or columns are missing, the app will show a helpful note and degrade gracefully instead of crashing.

## Preprocessing Logic
The app automatically:
- loads every CSV in `data/`
- standardizes column names to `snake_case`
- detects likely incident IDs, report IDs, dates, titles, descriptions, URLs, domains, country/location columns, and risk labels
- extracts year fields from usable date columns
- extracts source domains from URLs when needed
- parses incident-to-report links from `incidents.csv`
- joins incidents to reports when possible
- joins incidents to classification files when possible
- creates aggregates for:
  - incidents per year
  - reports per year
  - risk categories by year
  - reports per incident
  - top source domains
  - country/location counts

## Dashboard Tabs
### 1. Overview
Shows the big picture:
- KPI cards
- dual-line trend for incidents vs reports
- rolling average trend
- filtered incident table

### 2. Risk Evolution
Shows how categories change over time:
- year × risk category heatmap
- stacked area composition chart
- bump/rank chart
- risk summary table

### 3. Attention & Sources
Shows inequality in public attention:
- top incidents by linked reports
- long-tail histogram of reports per incident
- top source domains
- Lorenz curve for concentration
- incident detail card

### 4. Geographic & Advanced View
Shows spatial and relational structure:
- world map / globe projection
- top countries side chart
- incident–risk–source network graph

## Visualization Choices
- **Dual-line chart**: distinguishes incident volume from reporting volume
- **Heatmap**: shows temporal shifts in risk visibility cleanly
- **Stacked area chart**: emphasizes composition rather than only totals
- **Lorenz curve**: makes attention inequality visible
- **World map**: reveals where incidents are located when metadata is available
- **Network graph**: links incidents to both risk categories and media sources

## Data Notes
- The incident database reflects **reported/documented incidents**, not all real-world incidents.
- **Report count is a visibility measure**, not a direct measure of harm severity.
- Geographic views depend on available location metadata and exclude missing/ambiguous locations.
- Classification coverage varies across MIT, GMF, and CSET files.

## Run Locally
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
shiny run app.py
```

Then open the local URL shown in the terminal.

## NLP Processing Pipeline
Build processed datasets for TF-IDF models, transformer fine-tuning, clustering, and dashboard analysis:

```bash
python process_data.py --data-dir data --output-dir processed_data --split 80/20
```

For a train/validation/test split:

```bash
python process_data.py --data-dir data --output-dir processed_data --split 70/15/15
```

The pipeline writes:
- `processed_data/report_level_processed.csv`
- `processed_data/incident_level_processed.csv`
- `processed_data/classification_ready.csv`
- `processed_data/classification_train.csv`
- `processed_data/classification_test.csv`
- `processed_data/classification_val.csv` when a validation split is requested

Splits are assigned back to `incident_id`, so all reports from the same incident stay in the same split. When one report is linked to multiple incidents, those incidents are split together to avoid duplicated source text across train/test.

## TF-IDF Baseline Model
Train an explainable TF-IDF + Logistic Regression classifier:

```bash
python train_tfidf_logreg.py --data-path processed_data/classification_ready.csv --target mit_risk_domain
```

Change the target with `--target`, or edit `TARGET_LABEL_COLUMN` near the top of `train_tfidf_logreg.py`.

The baseline saves a `.joblib` model, metrics JSON, classification report CSV, test predictions CSV, and confusion matrix PNG under `model_outputs/tfidf_logreg_<target>/`.

## Customization Notes
You can safely modify:
- chart logic in [src/visualizations.py](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/src/visualizations.py)
- column detection and joins in [src/preprocess.py](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/src/preprocess.py)
- styling in [assets/styles.css](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/assets/styles.css)

If you want to add more advanced analytics later, the easiest next extensions are:
- animated timeline views
- TF-IDF keyword/topic explorer
- a refined geographic layer with manual geocoding or cleaned location mapping
