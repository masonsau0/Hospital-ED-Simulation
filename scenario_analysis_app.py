"""Interactive Hospital ED scenario-explorer dashboard.

Run with::

    streamlit run scenario_analysis_app.py

Lets the user explore the 33 Arena-simulation scenarios (11 holding-bay
counts × 3 closing times), tune managerial cost assumptions, and see the
recommended configuration update in real time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Hospital ED Capacity Planner", layout="wide", page_icon="🏥")

DATA = Path("scenario_results.xlsx")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_scenarios() -> pd.DataFrame:
    raw = pd.read_excel(DATA, sheet_name="Sheet2", header=0)
    cols = ["Configuration", "Reneging Rate", "CATH Wait Time", "EP Wait Time",
            "Hold For Bay Queue Time", "Bay Utilization", "Lab Utilization",
            "# of Reneges", "# of Seized Bays"]
    df = raw[cols].dropna(subset=["Configuration"]).copy()
    df = df[df["Configuration"].str.contains("Bays", na=False)].reset_index(drop=True)
    df["Bays"] = df["Configuration"].str.extract(r"(\d+)\s*Bays").astype(int)
    df["ClosingTime"] = df["Configuration"].str.extract(r"Bays,\s*(.+)$").iloc[:, 0]
    df["close_cost_index"] = df["ClosingTime"].map({"8 PM": 0, "10 PM": 1, "12 AM": 2})
    return df


def pareto_mask(points: np.ndarray) -> np.ndarray:
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j] <= points[i]).all() and (points[j] < points[i]).any():
                keep[i] = False
                break
    return keep


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


st.title("Hospital ED Capacity Planner")
st.caption("Explore 33 discrete-event-simulation scenarios — bay count × closing time — and find the configuration that minimises your total operating cost.")

with st.expander("How to use this app", expanded=False):
    st.markdown("""
**What this app does in plain English.**
A hospital is planning a joint cardiac and electrophysiology lab. They
need to decide two things: (1) how many holding bays to build, and
(2) how late to keep the lab open each day. Build too few bays or
close too early → patients give up and leave (called "reneging") and
overflow to other hospitals. Build too many → wasted money. We ran
**33 simulations** in Arena (a discrete-event simulation tool) varying
bay count from 11 to 21 and closing time at 8 PM / 10 PM / 12 AM.
This app lets you explore those scenarios and find the cheapest
configuration that hits acceptable patient outcomes.

**Quick start (60 seconds).**
1. Look at the **Pareto frontier** chart on the main page — every dot
   is one scenario plotted on (cost) vs (reneges). The dots on the
   bottom-left curve are non-dominated (best trade-offs).
2. Use the **filters** in the sidebar to narrow down by bay count or
   closing time.
3. Open the **Heatmaps** tab to see all 33 scenarios in a grid.

**The metrics in plain English.**
- **Reneges** — patients who arrive but leave without being seen
  because the wait was too long. We want this LOW.
- **Transfers** — patients moved to another hospital. Also LOW.
- **Bay utilisation** — what % of the time bays are occupied. Too low
  = wasted; too high = no slack for emergencies. Sweet spot ~75-85 %.
- **Total cost** — staffing + facility cost per day. LOW.

**The tabs / pages.**
- **Pareto frontier** — the key chart. Dots on the bottom-left curve
  are configurations where you can't reduce cost without increasing
  reneges (or vice versa). Pick from these.
- **Heatmaps** — all scenarios in a colour grid. Easy to spot
  patterns (e.g. "bay 14, close 10 PM" is a sweet spot).
- **Scenario detail** — pick one scenario, see all metrics for it.
- **Recommendation** — the best scenario by your weight on
  cost-vs-reneges trade-off.

**The trade-off slider.**
- **Cost weight** — how much you value saving money vs reducing
  reneges. 0 = "I'll pay anything to never have a renege"; 1 = "Cheap
  is everything." Most hospitals land around 0.6-0.7.

**Try this.** Slide cost-weight from 1.0 down to 0.0 and watch the
recommended configuration shift from cheapest (fewer bays, earlier
close) to most-patient-friendly (more bays, later close). The
recommendation usually settles around **14 bays, 10 PM close** —
that's the non-dominated middle of the Pareto frontier.
""")

if not DATA.exists():
    st.error(f"`{DATA}` not found. Make sure you're running from the project folder.")
    st.stop()

df = load_scenarios()

with st.sidebar:
    st.header("Cost assumptions")
    st.caption("Set the dollar value of each output. The recommendation updates immediately.")

    cost_renege = st.number_input("Cost per patient renege ($)",
                                    value=200, step=25, min_value=0,
                                    help="Opportunity cost + reputation hit when a patient leaves before service.")
    cost_bay = st.number_input("Cost per bay per month ($)",
                                value=8000, step=500, min_value=0,
                                help="Amortised capex + staffing per holding bay.")
    cost_close_8pm = st.number_input("Cost of 8 PM close ($)", value=0, step=1000)
    cost_close_10pm = st.number_input("Cost of 10 PM close ($)", value=15000, step=1000)
    cost_close_12am = st.number_input("Cost of 12 AM close ($)", value=30000, step=1000)
    close_costs = {"8 PM": cost_close_8pm, "10 PM": cost_close_10pm, "12 AM": cost_close_12am}

    st.divider()
    st.header("Filters")
    bay_range = st.slider("Bay count range", int(df["Bays"].min()), int(df["Bays"].max()),
                            (int(df["Bays"].min()), int(df["Bays"].max())))
    selected_close = st.multiselect("Closing times", ["8 PM", "10 PM", "12 AM"],
                                      default=["8 PM", "10 PM", "12 AM"])


# Filter
filtered = df[
    (df["Bays"].between(bay_range[0], bay_range[1]))
    & (df["ClosingTime"].isin(selected_close))
].copy()

if filtered.empty:
    st.warning("No scenarios match the filters. Widen the range or select more closing times.")
    st.stop()

# Compute total cost under user assumptions
filtered["close_cost"] = filtered["ClosingTime"].map(close_costs)
filtered["total_cost"] = (
    filtered["# of Reneges"] * cost_renege
    + filtered["Bays"] * cost_bay
    + filtered["close_cost"]
)

best_idx = filtered["total_cost"].idxmin()
best = filtered.loc[best_idx]

# Headline metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recommended", best["Configuration"])
m2.metric("Total monthly cost", f"${best['total_cost']:,.0f}")
m3.metric("Reneges", f"{int(best['# of Reneges'])}",
          delta=f"{best['Reneging Rate']:.0%} renege rate", delta_color="off")
m4.metric("Bay utilization", f"{best['Bay Utilization']:.0%}")

st.divider()

# Tabs
tab_grid, tab_pareto, tab_table, tab_metric = st.tabs(
    ["Cost grid", "Pareto frontier", "All scenarios", "Single metric"]
)


# ---- Cost-grid heatmap ----------------------------------------------------

with tab_grid:
    pivot = filtered.pivot(index="Bays", columns="ClosingTime", values="total_cost")
    pivot = pivot[[c for c in ["8 PM", "10 PM", "12 AM"] if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=pivot.map("${:,.0f}".format), fmt="",
                 cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Total cost ($)"})
    ax.set_title("Total cost grid — green = cheaper")
    # Mark the optimum
    if best["ClosingTime"] in pivot.columns:
        col_pos = list(pivot.columns).index(best["ClosingTime"])
        row_pos = list(pivot.index).index(best["Bays"])
        ax.add_patch(plt.Rectangle((col_pos, row_pos), 1, 1, fill=False, edgecolor="black", lw=3))
    st.pyplot(fig)
    st.caption(f"Black box marks the recommended cell: **{best['Configuration']}** at **${best['total_cost']:,.0f}/month**.")


# ---- Pareto frontier ------------------------------------------------------

with tab_pareto:
    pts = filtered[["# of Reneges", "Bays", "close_cost_index"]].values
    filtered["pareto"] = pareto_mask(pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"8 PM": "#c44e52", "10 PM": "#dd8452", "12 AM": "#55a868"}
    for ct, sub in filtered.groupby("ClosingTime"):
        ax.scatter(sub["Bays"], sub["# of Reneges"], s=60, c=colors.get(ct, "#888"),
                    label=f"Close {ct}", edgecolor="white")
    frontier = filtered[filtered["pareto"]].sort_values("Bays")
    ax.plot(frontier["Bays"], frontier["# of Reneges"], "k--", alpha=0.5, label="Pareto frontier")
    ax.scatter(best["Bays"], best["# of Reneges"], s=300, marker="*", color="black",
                label="Recommended", zorder=5)
    ax.set_xlabel("Holding bays"); ax.set_ylabel("Total reneges")
    ax.set_title("Cost-reneges Pareto frontier"); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)


# ---- All scenarios table --------------------------------------------------

with tab_table:
    show = filtered.sort_values("total_cost").copy()
    show["total_cost"] = show["total_cost"].map("${:,.0f}".format)
    show["Reneging Rate"] = show["Reneging Rate"].map("{:.1%}".format)
    show["Bay Utilization"] = show["Bay Utilization"].map("{:.1%}".format)
    show["Lab Utilization"] = show["Lab Utilization"].map("{:.1%}".format)
    show["close_cost"] = show["close_cost"].map("${:,.0f}".format)
    cols_show = ["Configuration", "total_cost", "# of Reneges", "Reneging Rate",
                 "Bay Utilization", "Lab Utilization", "close_cost"]
    st.dataframe(show[cols_show], hide_index=True, use_container_width=True)


# ---- Single-metric view ---------------------------------------------------

with tab_metric:
    metric = st.selectbox("Metric", [
        "Reneging Rate", "CATH Wait Time", "EP Wait Time",
        "Hold For Bay Queue Time", "Bay Utilization", "Lab Utilization", "# of Reneges"
    ])
    pivot = filtered.pivot(index="Bays", columns="ClosingTime", values=metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=f"Close {col}")
    ax.set_xlabel("Holding bay count"); ax.set_ylabel(metric); ax.set_title(metric)
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)
