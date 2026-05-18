# Project 2: AI Incident Data Visualization

## Project Description
This project analyzes an AI incident dataset to understand how the nature of AI-related harm has changed over time. The dataset contains incident-level records, linked source reports, and several classification files that describe each incident from different perspectives such as risk type, intent, technical failure mode, sector, geography, and harm severity.

The central goal of the project is to build a visual story around this question:

**How has AI harm evolved from isolated technical failures into broader social, political, and intentional misuse?**

The working thesis for the project is:

**Since 2023, the center of gravity of AI harm has shifted from accidental system failures toward intentional misuse, misinformation, and socially scaled harms.**

## Dataset Overview
The project uses the files in the [data](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data) folder.

Main files:
- [incidents.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/incidents.csv): core incident table with one row per incident
- [reports.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/reports.csv): linked articles and source reports for each incident
- [classifications_MIT.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/classifications_MIT.csv): high-level risk taxonomy, timing, and intent
- [classifications_GMF.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/classifications_GMF.csv): AI goals, technologies, and technical failure modes
- [classifications_CSETv1.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/classifications_CSETv1.csv): detailed harm, sector, autonomy, geography, and public sector labels

Supporting files:
- [duplicates.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/duplicates.csv): duplicate incident mapping
- [submissions.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/submissions.csv): incoming submissions
- [quickadd.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/quickadd.csv): candidate records

## Project Objective
The objective is to produce a clear visual narrative that explains:
- how AI incidents have grown over time
- how the dominant categories of harm have shifted
- how intentional misuse compares with unintentional failure
- which technical failure modes are most associated with recent incidents
- which cases receive the most public and media attention

## Initial Plan
### 1. Understand and prepare the data
- Inspect the structure of the main incident, report, and classification files
- Clean join keys between `incident_id` and `Incident ID`
- Parse linked report IDs from the incident table
- Check duplicates, missing values, and coverage across classification tables

### 2. Build the core analysis tables
- Create an incident-level analysis table from [incidents.csv](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/data/incidents.csv)
- Join MIT classifications for risk domain, timing, and intent
- Join GMF classifications for technical goals and failure modes
- Join CSET classifications for harm context, sector, and severity

### 3. Develop the main story visuals
- Chart incident growth over time
- Show risk-domain shifts by year
- Compare intentional vs unintentional incidents
- Visualize technical failure modes behind recent harms
- Highlight the most-covered incidents using linked report counts

### 4. Refine the narrative
- Identify the 3-5 strongest insights
- Remove weak or redundant charts
- Focus the final presentation on one central argument supported by a small number of strong visuals

### 5. Deliver final outputs
- A polished notebook with step-by-step analysis
- Final charts for presentation or dashboard use
- A short written narrative explaining the main findings

## Current Working Story
The most promising story direction is:

**AI incidents are no longer dominated only by technical malfunctions. In recent years, the dataset shows a stronger pattern of deliberate misuse, misinformation, deepfakes, fraud, and other socially scaled harms.**

## Project Files
- Draft analysis notebook: [ai_incidents_story_draft.ipynb](/Users/dienmayhaituyet/Documents/Project2_DataVisualization/notebooks/ai_incidents_story_draft.ipynb)
