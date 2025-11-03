"""
Generate charts that compare NATO military spending to key adversaries.

The module extracts the logic that originally lived inside the
`AnalyzeAndPlotSpending.ipynb` notebook and turns it into reusable, composable
functions.  Execute the module as a script to regenerate all published figures:

    python -m analyze_and_plot_spending --plots all
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import MultiPolygon, box


# ------------------------------------------------------------------------------
# Paths & lightweight configuration
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "cache"
FIGURES_DIR = REPO_ROOT / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Domain constants
# ------------------------------------------------------------------------------
NATO_MEMBERS = [
    "Albania",
    "Belgium",
    "Bulgaria",
    "Canada",
    "Croatia",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Türkiye",
    "United Kingdom",
    "United States of America",
]

# Keep a European subset (US + Canada excluded) for several plots
EUROPEAN_NATO = [
    c
    for c in NATO_MEMBERS
    if c not in {"Canada", "United States of America"}
]

ADVERSARIES = ["Russia", "China"]

# Mapping between SIPRI dataset names and Natural Earth shapefile names
NAME_FIX: Mapping[str, str] = {
    "Congo, DR": "Democratic Republic of the Congo",
    "Congo, Republic": "Republic of the Congo",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Iran (Islamic Republic of)": "Iran",
    "Korea, North": "North Korea",
    "Korea, South": "South Korea",
    "Lao PDR": "Laos",
    "Micronesia (Federated States of)": "Federated States of Micronesia",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "Timor-Leste": "East Timor",
    "Türkiye": "Turkey",
    "Viet Nam": "Vietnam",
}


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _clean_numeric(series: pd.Series, strip_percent: bool = False) -> pd.Series:
    """Convert SIPRI or OECD numeric columns to floats, handling sentinels."""

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(". .", "", regex=False)
        .str.replace("...", "", regex=False)
        .str.replace("xxx", "", regex=False)
        .str.replace("..", "", regex=False)
        .str.replace("†", "", regex=False)
        .str.replace("\u2013", "-", regex=False)  # ensure en dash becomes hyphen
    )
    if strip_percent:
        cleaned = cleaned.str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _ensure_natural_earth(scale: str) -> gpd.GeoDataFrame:
    """
    Download and cache the Natural Earth Admin-0 dataset for the requested
    resolution (e.g., '50m', '110m').
    """

    base_name = f"ne_{scale}_admin_0_countries"
    zip_path = CACHE_DIR / f"{base_name}.zip"
    extract_dir = CACHE_DIR / base_name

    if not extract_dir.exists():
        if not zip_path.exists():
            url = f"https://naciscdn.org/naturalearth/{scale}/cultural/{base_name}.zip"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    shp_path = extract_dir / f"{base_name}.shp"
    return gpd.read_file(shp_path)


def set_pubstyle_econ_ft(base_fontsize: float = 12.0) -> None:
    """Matplotlib style tuned to look like an Economist / FT chart."""

    mpl.rcParams.update(
        {
            "font.family": ["IBM Plex Sans", "Source Sans 3", "Lato", "DejaVu Sans"],
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize * 1.6,
            "axes.labelsize": base_fontsize * 1.25,
            "legend.fontsize": base_fontsize * 0.95,
            "xtick.labelsize": base_fontsize * 1.0,
            "ytick.labelsize": base_fontsize * 1.0,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "legend.loc": "upper left",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
        }
    )


def economist_axis_touches(ax: plt.Axes) -> None:
    """Bring axis spines in line with the house style."""

    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_color("#3a3a3a")
    ax.spines["bottom"].set_color("#3a3a3a")
    ax.grid(axis="y", color="#8a8a8a")
    ax.grid(axis="x", visible=False)


def add_caption(fig: plt.Figure, text: str, y_offset: float = 0.02) -> None:
    """Place a small-caption / source note underneath the figure."""

    fig.text(0.0, y_offset, text, ha="left", va="bottom", fontsize=10, color="#5a5a5a")


# ------------------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------------------
def load_constant_spending(years: Sequence[int] = (2022, 2023, 2024)) -> pd.DataFrame:
    """Load SIPRI constant 2023 USD military spending for the requested years."""

    filepath = DATA_RAW / "military_spending_constant_2023.csv"
    df = pd.read_csv(filepath, sep=",", skiprows=5)

    keep_cols = ["Country", *map(str, years)]
    df = df[keep_cols].copy()
    for year in years:
        col = str(year)
        df[col] = _clean_numeric(df[col])
    return df.dropna(subset=["Country"])


def load_share_of_gdp() -> pd.DataFrame:
    """Load SIPRI GDP-share data and return a tidy (long-form) DataFrame."""

    filepath = DATA_RAW / "military_expenditures_in_shared_of_gdp.tsv"
    df = pd.read_csv(filepath, sep="\t", skiprows=5)
    value_cols = [c for c in df.columns if c.isdigit()]

    tidy = df.melt(
        id_vars=["Country"],
        value_vars=value_cols,
        var_name="Year",
        value_name="Spending",
    )
    tidy["Year"] = tidy["Year"].astype(int)
    tidy["Spending"] = _clean_numeric(tidy["Spending"], strip_percent=True)

    # SIPRI uses both spellings across datasets; normalise once here.
    tidy["Country"] = tidy["Country"].replace({"Turkey": "Türkiye"})

    return tidy.dropna(subset=["Spending"])


def load_social_spending() -> pd.DataFrame:
    """Load OECD social expenditure data in tidy form (percentage of GDP)."""

    filepath = DATA_RAW / "social_spending_ssocx.csv"
    df = pd.read_csv(filepath)

    trimmed = (
        df[["Reference area", "TIME_PERIOD", "OBS_VALUE"]]
        .rename(
            columns={
                "Reference area": "Country",
                "TIME_PERIOD": "Year",
                "OBS_VALUE": "SocialSpendingPctGDP",
            }
        )
        .dropna(subset=["SocialSpendingPctGDP"])
    )

    trimmed["Year"] = trimmed["Year"].astype(int)
    trimmed["SocialSpendingPctGDP"] = (
        _clean_numeric(trimmed["SocialSpendingPctGDP"]).astype(float)
    )

    # Align naming with the SIPRI data (drop " of America" for readability).
    trimmed["Country"] = trimmed["Country"].replace(
        {"United States": "United States of America"}
    )

    return trimmed.dropna(subset=["SocialSpendingPctGDP"])


# ------------------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------------------
def _largest_polygon_point(geom) -> tuple[float, float]:
    """Return a robust label point inside a polygon/multipolygon."""

    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    pt = geom.representative_point()
    return float(pt.x), float(pt.y)


def _declutter_labels(
    df: gpd.GeoDataFrame, min_dist: float, sort_key: str
) -> gpd.GeoDataFrame:
    """Greedy decluttering: keep the largest movers separated on the map."""

    kept = []
    ordered = df.reindex(df[sort_key].abs().sort_values(ascending=False).index)
    for idx, row in ordered.iterrows():
        x, y = row["label_x"], row["label_y"]
        if all((x - ordered.loc[j, "label_x"]) ** 2 + (y - ordered.loc[j, "label_y"]) ** 2 >= min_dist**2 for j in kept):
            kept.append(idx)
    return ordered.loc[kept]


# ------------------------------------------------------------------------------
# Plot functions
# ------------------------------------------------------------------------------
def plot_world_percent_change(spending: pd.DataFrame) -> Path:
    """
    Plot worldwide % change in military spending between 2022 and 2024,
    using a diverging colormap centred on zero.
    """

    world = _ensure_natural_earth("50m")
    data = spending[["Country", "2022", "2024"]].copy()
    data = data.dropna(subset=["2022", "2024"])
    data["pct_increase"] = (data["2024"] - data["2022"]) / data["2022"] * 100

    data["ne_name"] = data["Country"].replace(NAME_FIX)

    merged = world.merge(
        data[["ne_name", "pct_increase"]],
        left_on="NAME",
        right_on="ne_name",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#e9f6ff")
    ax.set_facecolor("#e9f6ff")
    merged.plot(ax=ax, facecolor="#f0f0f0", edgecolor="white", linewidth=0.4, zorder=1)

    available = merged.dropna(subset=["pct_increase"]).copy()
    vmin, vmax = available["pct_increase"].min(), available["pct_increase"].max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdYlGn

    available.plot(
        ax=ax,
        column="pct_increase",
        cmap=cmap,
        linewidth=0.4,
        edgecolor="white",
        norm=norm,
        zorder=2,
    )
    available.boundary.plot(ax=ax, linewidth=0.2, edgecolor="#444444", zorder=3)

    coords = [ _largest_polygon_point(geom) for geom in available.geometry ]
    if coords:
        available["label_x"] = [pt[0] for pt in coords]
        available["label_y"] = [pt[1] for pt in coords]
    else:
        available["label_x"] = []
        available["label_y"] = []
    labels = _declutter_labels(available, min_dist=5.0, sort_key="pct_increase")

    for _, row in labels.iterrows():
        ax.text(
            row["label_x"],
            row["label_y"],
            f"{row['pct_increase']:+.0f}%",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#111111",
            zorder=4,
            clip_on=True,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)],
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Military spending % change (2022 → 2024)", rotation=90, labelpad=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))

    ax.set_title(
        "Worldwide % Change in Military Spending (2022 → 2024, constant 2023 USD)",
        fontsize=16,
        loc="left",
        weight="bold",
    )
    ax.text(
        0.01,
        0.01,
        "Basemap: Natural Earth 1:50m · Projection: WGS84 · Source: SIPRI",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
    )

    ax.set_axis_off()
    plt.tight_layout()

    output = FIGURES_DIR / "world_increase_military_spending.png"
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_europe_nato_change(spending: pd.DataFrame) -> Path:
    """Choropleth: NATO members in Europe and Türkiye, 2022→2024 % change."""

    world = _ensure_natural_earth("50m")
    europe = world[(world["CONTINENT"] == "Europe") | (world["ADM0_A3"] == "TUR")].copy()
    name_col = "NAME_EN" if "NAME_EN" in europe.columns else "NAME"
    europe["NAME_JOIN"] = europe[name_col].fillna(europe["NAME"])

    data = spending[["Country", "2022", "2024"]].copy()
    data = data[data["Country"].isin(NATO_MEMBERS)]
    data["pct_increase"] = (data["2024"] - data["2022"]) / data["2022"] * 100
    data["ne_name"] = data["Country"].replace({"Türkiye": "Turkey", "Türkiye (Turkey)": "Turkey"})

    europe_3035 = europe.to_crs(3035)
    merged = europe_3035.merge(
        data[["ne_name", "pct_increase"]],
        left_on="NAME_JOIN",
        right_on="ne_name",
        how="left",
    )

    minx, miny, maxx, maxy = 1_800_000, 1_000_000, 7_200_000, 4_800_000
    clip_box = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=merged.crs)

    outline = gpd.overlay(europe_3035, clip_box, how="intersection")
    clipped = gpd.overlay(merged, clip_box, how="intersection")

    coords = [ _largest_polygon_point(geom) for geom in clipped.geometry ]
    if coords:
        clipped["label_x"] = [pt[0] for pt in coords]
        clipped["label_y"] = [pt[1] for pt in coords]
    else:
        clipped["label_x"] = []
        clipped["label_y"] = []

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.rcParams.update({"figure.facecolor": "#a6cee3"})
    ax.set_facecolor("#a6cee3")

    outline.plot(ax=ax, facecolor="#f1f1f1", edgecolor="white", linewidth=0.6, zorder=1)

    countries = clipped.dropna(subset=["pct_increase"]).copy()
    countries.plot(
        ax=ax,
        column="pct_increase",
        cmap="YlOrRd",
        linewidth=0.7,
        edgecolor="white",
        zorder=2,
    )
    countries.boundary.plot(ax=ax, linewidth=0.2, edgecolor="#444444", zorder=3)

    for _, row in countries.iterrows():
        ax.text(
            row["label_x"],
            row["label_y"],
            f"{row['pct_increase']:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1a1a1a",
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")],
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=countries["pct_increase"].min(), vmax=countries["pct_increase"].max()),
        cmap="YlOrRd",
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label("Military spending % change (2022 → 2024)", rotation=90, labelpad=12)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))

    ax.set_title(
        "European NATO: Military Spending Change (2022 → 2024, constant 2023 USD)",
        fontsize=15,
        pad=10,
        loc="left",
        weight="bold",
    )
    ax.text(
        0.01,
        0.02,
        "Projection: EPSG:3035 (LAEA Europe) · Basemap: Natural Earth 1:50m",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
    plt.tight_layout()

    output = FIGURES_DIR / "europe_turkiye_military_spending_2.png"
    fig.savefig(output, dpi=1000)
    plt.close(fig)
    return output


def plot_us_vs_top_n(spending: pd.DataFrame, top_n: int = 8) -> Path:
    """Bar chart: United States vs the next N biggest spenders."""

    year = "2024"
    data = spending[["Country", year]].dropna()
    data = data[data[year] > 0]

    usa_value = float(
        data.loc[data["Country"] == "United States of America", year].squeeze()
    )
    others = (
        data[data["Country"] != "United States of America"]
        .nlargest(top_n, columns=year)
        .copy()
    )
    others["billions"] = others[year] / 1_000.0
    usa_billions = usa_value / 1_000.0

    needed_countries = others["Country"].tolist()
    needed_spends = others["billions"].tolist()

    set_pubstyle_econ_ft(base_fontsize=12)
    fig = plt.figure(figsize=(10, 6.2), facecolor="#f9f4ef")
    ax = plt.gca()
    ax.set_facecolor("#f9f4ef")

    economist_axis_touches(ax)
    ax.set_ylim(0, max(usa_billions, sum(needed_spends)) * 1.15)

    ax.set_xlabel("Category")
    ax.set_ylabel("Military spending (constant 2023 USD, billions)")
    ax.set_title(
        f"Military spending: United States vs Next {top_n} Countries ({year})",
        loc="left",
        fontweight="bold",
    )
    subtitle = "Data in constant 2023 US dollars, converted from SIPRI millions."
    fig.suptitle(subtitle, x=0.06, y=0.94, ha="left", fontsize=12, color="#5a5a5a")

    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1]))

    color_map = {
        "Russia": "#333333",
        "China": "#FF7F0E",
        "United Kingdom": "#17becf",
        "France": "#0055A4",
        "Germany": "#8c564b",
        "Saudi Arabia": "#006C35",
        "India": "#9467BD",
        "Ukraine": "#FFD700",
    }

    ax.bar(0, usa_billions, zorder=3, color="#1F77B4")
    ax.text(0, usa_billions / 2, "United States", ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    bottom = 0.0
    for country, value in zip(needed_countries, needed_spends):
        ax.bar(
            1,
            value,
            bottom=bottom,
            zorder=3,
            color=color_map.get(country, "#7f7f7f"),
        )
        ax.text(
            1,
            bottom + value / 2,
            country,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        bottom += value

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["United States", f"Next {top_n}\ncountries"])

    add_caption(fig, "Source: SIPRI (constant 2023 USD)")

    plt.tight_layout(rect=[0, 0, 0.86, 0.95])
    output = FIGURES_DIR / "total_military_spending_comparison.png"
    fig.savefig(output, dpi=1000)
    plt.close(fig)
    return output


def _prepare_nato_timeseries(percent_gdp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return helper DataFrames used across the NATO time-series charts."""

    nato = percent_gdp[percent_gdp["Country"].isin(NATO_MEMBERS)].copy()
    nato_long = nato.groupby(["Country", "Year"], as_index=False)["Spending"].mean()

    result = {
        "nato_long": nato_long,
        "us": nato_long[nato_long["Country"] == "United States of America"],
        "canada": nato_long[nato_long["Country"] == "Canada"],
        "europe": nato_long[nato_long["Country"].isin(EUROPEAN_NATO)],
    }

    for adversary in ADVERSARIES:
        result[adversary.lower()] = percent_gdp[
            percent_gdp["Country"] == adversary
        ].groupby("Year", as_index=False)["Spending"].mean()

    return result


def plot_nato_vs_adversaries(percent_gdp: pd.DataFrame) -> Path:
    """Time-series comparing NATO averages to Russia and China (% GDP)."""

    data = _prepare_nato_timeseries(percent_gdp)

    europe_avg = (
        data["europe"].groupby("Year")["Spending"].mean().rename("Europe").to_frame()
    )
    canada_avg = data["canada"].rename(columns={"Spending": "Canada"})
    us_avg = data["us"].rename(columns={"Spending": "United States"})
    russia_avg = data["russia"].rename(columns={"Spending": "Russia"})
    china_avg = data["china"].rename(columns={"Spending": "China"})

    set_pubstyle_econ_ft(base_fontsize=12)
    fig = plt.figure(figsize=(10, 6.2), facecolor="#f9f4ef")
    ax = plt.gca()
    ax.set_facecolor("#f9f4ef")

    ax.plot(europe_avg.index, europe_avg["Europe"], linewidth=2.6, color="#9467BD", label="European NATO")
    ax.plot(canada_avg["Year"], canada_avg["Canada"], linewidth=2.6, linestyle="--", color="#D62728", label="Canada")
    ax.plot(us_avg["Year"], us_avg["United States"], linewidth=2.6, linestyle="-.", color="#1F77B4", label="United States")
    ax.plot(russia_avg["Year"], russia_avg["Russia"], linewidth=2.6, color="#333333", label="Russia")
    ax.plot(china_avg["Year"], china_avg["China"], linewidth=2.6, color="#FF7F0E", label="China")

    ax.axhline(2, color="#0c0c0c", linestyle=(0, (3, 2)), linewidth=1.8, alpha=0.9)
    ax.text(1950, 1.3, "NATO target (2%)", fontsize=10, va="bottom", ha="left", color="#000000", fontweight="bold")

    ax.set_ylim(0, 15)

    events = [
        (1991, "Collapse of USSR"),
        (2001, "US invades Afghanistan"),
        (2014, "Russia in Ukraine (Crimea/Donbas)"),
        (2022, "Russia invades Ukraine"),
    ]
    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.06 * (ymax - ymin) + 5

    for x, label in events:
        ax.axvline(x=x, color="#2f2f2f", linestyle=(0, (2, 3)), linewidth=1.6, alpha=0.65, zorder=1)
        ax.text(
            x - 1.5,
            label_y,
            label,
            rotation=90,
            fontsize=9.5,
            va="bottom",
            ha="left",
            color="#2f2f2f",
            fontweight="bold",
        )

    economist_axis_touches(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Military spending (% of GDP)")
    ax.set_title(
        "Military Spending as % of GDP",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    fig.suptitle(
        "NATO vs Russia & China, 1949–2024; NATO target highlighted",
        x=0.06,
        y=0.90,
        ha="left",
        fontsize=12,
        color="#5a5a5a",
    )

    legend = ax.legend(
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=1.0,
        loc="upper left",
        fontsize=10,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.2)

    ax.margins(x=0.01)
    ax.tick_params(axis="both", which="major", labelsize=11.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=len(ax.get_xticks()) * 2))
    ax.tick_params(axis="x", rotation=45)

    add_caption(fig, "Source: SIPRI; averages are unweighted unless noted.")
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    output = FIGURES_DIR / "countries_military_spending.png"
    fig.savefig(output, dpi=1000)
    plt.close(fig)
    return output


def plot_europe_two_percent(percent_gdp: pd.DataFrame) -> Path:
    """Share of European NATO countries hitting the 2% guideline per year."""

    data = _prepare_nato_timeseries(percent_gdp)
    europe = data["europe"]

    summary = (
        europe.groupby("Year")["Spending"]
        .agg(total=lambda s: s.notna().sum(), hits=lambda s: (s >= 2).sum())
        .reset_index()
    )
    summary["pct_above_2"] = summary["hits"] / summary["total"] * 100

    set_pubstyle_econ_ft(base_fontsize=12)
    fig = plt.figure(figsize=(10, 6.2), facecolor="#f9f4ef")
    ax = plt.gca()
    ax.set_facecolor("#f9f4ef")

    ax.plot(
        summary["Year"],
        summary["pct_above_2"],
        linewidth=3,
        marker="o",
        markersize=4,
        color="#9467BD",
        label="European NATO ≥ 2%",
    )

    events = [
        (1991, "Collapse of USSR"),
        (2002, "Prague Summit: Setting 2% Target"),
        (2014, "Russia in Ukraine (Crimea/Donbas)"),
        (2022, "Russia invades Ukraine"),
    ]
    for x, _ in events:
        ax.axvline(x=x, color="black", linestyle="--", linewidth=2.2, alpha=0.85, zorder=1)

    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.1 * (ymax - ymin) + 25

    for x, label in events:
        ax.text(
            x - 1.75,
            label_y,
            label,
            rotation=90,
            fontsize=9.5,
            va="bottom",
            ha="left",
            color="#2f2f2f",
            fontweight="bold",
        )

    economist_axis_touches(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of countries (%)")
    ax.set_title(
        "Share of European NATO Countries Meeting the 2% Spending Target",
        loc="left",
        fontweight="bold",
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=len(ax.get_xticks()) * 2))
    ax.tick_params(axis="x", rotation=45, labelsize=11.5)
    ax.legend().remove()

    add_caption(fig, "Source: NATO and SIPRI")
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    output = FIGURES_DIR / "european_countries_meeting_2_percent_target.png"
    fig.savefig(output, dpi=1000)
    plt.close(fig)
    return output


def plot_social_spending(social: pd.DataFrame) -> Path:
    """Compare NATO social expenditure (% GDP) for US, Canada, and Europe."""

    filtered = social[
        social["Country"].isin(
            {"United States of America", "Canada", *EUROPEAN_NATO}
        )
    ].copy()
    filtered = filtered[filtered["Year"] >= 1949]

    europe = filtered[
        ~filtered["Country"].isin({"United States of America", "Canada"})
    ]
    europe_avg = europe.groupby("Year")["SocialSpendingPctGDP"].mean().reset_index()

    us = (
        filtered[filtered["Country"] == "United States of America"]
        .sort_values("Year")
        .reset_index(drop=True)
    )
    canada = (
        filtered[filtered["Country"] == "Canada"]
        .sort_values("Year")
        .reset_index(drop=True)
    )

    set_pubstyle_econ_ft(base_fontsize=12)
    fig = plt.figure(figsize=(10, 6.2), facecolor="#f9f4ef")
    ax = plt.gca()
    ax.set_facecolor("#f9f4ef")

    ax.plot(us["Year"], us["SocialSpendingPctGDP"], label="United States", color="#1F77B4", linestyle="--", linewidth=4)
    ax.plot(canada["Year"], canada["SocialSpendingPctGDP"], label="Canada", color="#D62728", linewidth=4)
    ax.plot(
        europe_avg["Year"],
        europe_avg["SocialSpendingPctGDP"],
        label="European NATO (avg)",
        color="#9467BD",
        linewidth=4,
        linestyle="-.",
    )

    events = [
        (1991, "Collapse of USSR"),
        (2008, "Great Recession"),
        (2020, "COVID-19 Pandemic"),
    ]
    for x, _ in events:
        ax.axvline(x=x, color="black", linestyle="--", linewidth=2.2, alpha=0.85, zorder=1)

    ymin, ymax = ax.get_ylim()
    base_y = ymin + 0.1 * (ymax - ymin)

    for x, label in events:
        offset = 20 if x != 2020 else 12.5
        ax.text(
            x - 1.0,
            base_y + offset,
            label,
            rotation=90,
            fontsize=9.5,
            va="bottom",
            ha="left",
            color="#2f2f2f",
            fontweight="bold",
        )

    ax.set_title("Public Social Spending as % of GDP", loc="left", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Social spending (% of GDP)")
    fig.suptitle(
        "Considers only public/government contribution and non-private spending.",
        x=0.06,
        y=0.90,
        ha="left",
        fontsize=12,
        color="#5a5a5a",
    )

    legend = ax.legend(
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=1.0,
        loc="upper left",
        fontsize=9,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.2)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.tick_params(axis="x", rotation=45)

    add_caption(fig, "Source: OECD SOCX")
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    output = FIGURES_DIR / "nato_countries_social_spending.png"
    fig.savefig(output, dpi=1000)
    plt.close(fig)
    return output


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
PLOT_REGISTRY = {
    "world-change": plot_world_percent_change,
    "europe-change": plot_europe_nato_change,
    "us-vs-top": plot_us_vs_top_n,
    "nato-vs-adversaries": plot_nato_vs_adversaries,
    "europe-two-percent": plot_europe_two_percent,
    "social-spending": plot_social_spending,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate military spending visualisations.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", *PLOT_REGISTRY.keys()],
        help="Which plots to generate (default: all).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="How many countries to stack against the US in the comparison plot.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> List[Path]:
    args = parse_args(argv)
    requested = list(PLOT_REGISTRY.keys()) if "all" in args.plots else args.plots

    spending = load_constant_spending()
    percent_gdp = load_share_of_gdp()
    social = load_social_spending()

    outputs: List[Path] = []
    for plot_key in requested:
        func = PLOT_REGISTRY[plot_key]
        if plot_key == "us-vs-top":
            output = func(spending=spending, top_n=args.top_n)
        elif plot_key in {"world-change", "europe-change"}:
            output = func(spending=spending)
        elif plot_key in {"nato-vs-adversaries", "europe-two-percent"}:
            output = func(percent_gdp=percent_gdp)
        elif plot_key == "social-spending":
            output = func(social=social)
        else:  # pragma: no cover - defensive in case of future additions
            raise ValueError(f"Unsupported plot key: {plot_key}")
        outputs.append(output)

    return outputs


if __name__ == "__main__":
    for path in main():
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
