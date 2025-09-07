# nato-and-adversaries-spending

Analyze military spending for NATO and strategic adversaries, with optional comparison to social spending. This repo contains data files and Jupyter notebooks to explore trends and produce plots.

## What's Here

- `AnalyzeAndPlotSpending.ipynb`: Load and visualize military spending.
- `SocialSpending.ipynb`: Explore social spending datasets (optional).
- `military_expenditures_in_shared_of_gdp.tsv`: Tab-separated military spending dataset.
- `SIPRI-Milex-data-1949-2024-all-countries.xlsx`: SIPRI military expenditure data (full source file).
- `social_spending_owid.csv`, `social_spending_ssocx.csv`, `social-expenditure-as-percentage-of-gdp.csv`: Social spending datasets.

## Quick Start

Prereqs: Python 3.9+, and Jupyter. Install the common libs:

```
pip install pandas seaborn matplotlib jupyter
```

Launch Jupyter:

```
jupyter notebook
```

Open `AnalyzeAndPlotSpending.ipynb` and run all cells.

## Data Notes

- The file `military_expenditures_in_shared_of_gdp.tsv` is tab-separated and contains 5 non-header lines at the top. Read it with either of the following:
  - Use `skiprows`: `pd.read_csv('military_expenditures_in_shared_of_gdp.tsv', sep='\t', skiprows=5)`
  - Or specify header row explicitly: `pd.read_csv('military_expenditures_in_shared_of_gdp.tsv', sep='\t', header=5)`
- SIPRI file is large; filter to countries/years of interest before plotting to keep notebooks responsive.

## Typical Workflow

1. Load military data for your target countries.
2. Clean/reshape as needed (e.g., set `Country` and `Year`, handle missing values).
3. Plot trends with `seaborn`/`matplotlib`.
4. Optionally merge with social spending data to compare shares of GDP.

## Reproducibility

- Pin your environment if desired (example):

```
pip install --require-virtualenv && python -m venv .venv && source .venv/bin/activate
pip install pandas==2.* seaborn==0.13.* matplotlib==3.* jupyter==1.*
```

## License and Attribution

- Data sources include SIPRI (military expenditure) and OWID/SSOCx (social spending). Please follow their licenses and attribution requirements.
- Code in this repo is provided as-is; add a license file if you intend to distribute.
