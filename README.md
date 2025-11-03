# nato-and-adversaries-spending

Military and social-spending analysis for NATO members and strategic adversaries.  
The original exploratory notebook has been extracted into a reusable script so the
figures can be reproduced without stepping through Jupyter.

## Repository Layout

- `src/analyze_and_plot_spending.py` – CLI for regenerating every chart.
- `notebooks/AnalyzeAndPlotSpending.ipynb` – exploratory workflow (now referencing the organised paths).
- `data/raw/` – source datasets from SIPRI and OECD; `data/cache/` holds Natural Earth shapefiles downloaded on demand.
- `figures/` – generated PNG outputs.
- `requirements.txt` – minimal runtime dependencies (pandas, geopandas, matplotlib, etc.).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Regenerate Figures from the CLI

All plots from the notebook can be rebuilt with one command:

```bash
python -m analyze_and_plot_spending --plots all
```

Use `--plots` to target specific outputs (options: `world-change`, `europe-change`,
`us-vs-top`, `nato-vs-adversaries`, `europe-two-percent`, `social-spending`) and
`--top-n` to adjust how many countries are stacked against the United States.
Charts are saved to `figures/`.

## Working in Jupyter

Launch Jupyter from the repository root so relative paths resolve correctly:

```bash
jupyter notebook notebooks/AnalyzeAndPlotSpending.ipynb
```

The notebook reads data from `../data/raw/` and saves exports into `../figures/`.

## Data Sources & Attribution

- SIPRI: `military_spending_constant_2023.csv` and `military_expenditures_in_shared_of_gdp.tsv`
- Natural Earth: Admin-0 boundaries (downloaded automatically into `data/cache/`)
- OECD SOCX: `social_spending_ssocx.csv`

Respect the licence terms of each provider when redistributing data or derived work.
