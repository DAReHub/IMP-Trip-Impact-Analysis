"""Trip-level comparison of a baseline and a scenario MATSim run.

Trips are grouped into three categories (public_transport, private_motorised, active)
from the 'modes' field. Results: per-category summary, mode shares, matched-trip effects
by baseline category, and a mode-shift matrix.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- settings
BASE_PATH = ""
SCEN_PATH = ""
OUT_DIR = Path("trip_results")
SCEN_NAME = "S1"

# Set True to report trips using both PT and a motorised mode as their own category
# instead of folding them into public_transport.
MULTIMODAL_SEPARATE = False

PT_MODES = {"pt", "bus", "rail", "train", "subway", "tram", "ferry"}
ACTIVE_MODES = {"walk", "non_network_walk", "bike", "bicycle", "cycle"}
MOTORISED_MODES = {"car", "car_passenger", "taxi", "ride", "motorcycle", "motorbike"}


# ---------------------------------------------------------------- helpers
def mode_group(modes):
    """Classify a trip's 'modes' string into one category.

    Walk access/egress is ignored when another mode is present.
    Mixed-trip priority: public transport > private motorised > active.
    """
    if pd.isna(modes):
        return "unknown"
    parts = {m for m in str(modes).split("-") if m}
    has_pt = bool(parts & PT_MODES)
    has_motor = bool(parts & MOTORISED_MODES)
    if has_pt and has_motor and MULTIMODAL_SEPARATE:
        return "multimodal"
    if has_pt:
        return "public_transport"
    if has_motor:
        return "private_motorised"
    if parts and parts <= ACTIVE_MODES:
        return "active"
    return "other"  # a mode name not listed in the sets above


def load_trips(path):
    t = pd.read_csv(path, sep=";")
    for col in ["dep_time", "trav_time", "wait_time"]:
        t[col] = pd.to_timedelta(t[col]).dt.total_seconds()  # hh:mm:ss -> seconds
    t["mode_group"] = t["modes"].map(mode_group)
    return t


def summary_by_category(t, categories):
    g = t.groupby("mode_group")
    out = pd.DataFrame(
        {
            "trips": g.size(),
            "person_hours": g["trav_time"].sum() / 3600,
            "mean_trip_min": g["trav_time"].mean() / 60,
            "mean_wait_min": g["wait_time"].mean() / 60,
            "km": g["traveled_distance"].sum() / 1000,
            "mean_detour_ratio": g["traveled_distance"].sum() / g["euclidean_distance"].sum(),
        }
    ).reindex(categories)  # categories absent from one run appear as NaN
    # An absent category means zero trips/hours/km (means stay NaN)
    sum_cols = ["trips", "person_hours", "km"]
    out[sum_cols] = out[sum_cols].fillna(0)
    return out


# ---------------------------------------------------------------- analysis
def analyse(base_path, scen_path, out_dir=OUT_DIR, scen_name=SCEN_NAME, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_trips(base_path)
    scen = load_trips(scen_path)

    # Check the raw mode chain -> category mapping (anything in 'other' needs adding to the sets)
    raw_to_group = (
        pd.concat([base[["modes", "mode_group"]], scen[["modes", "mode_group"]]])
        .drop_duplicates()
        .sort_values(["mode_group", "modes"])
    )
    categories = sorted(set(base["mode_group"]) | set(scen["mode_group"]))
    if verbose:
        print("Categories found:", categories)
        print(raw_to_group.to_string(index=False))

    # ---- 1. Per-category summary ----
    sb = summary_by_category(base, categories)
    ss = summary_by_category(scen, categories)
    change = ss - sb
    pct = change / sb.where(sb != 0) * 100
    per_category = pd.concat({"base": sb, "scen": ss, "change": change, "pct": pct}, axis=1)

    # ---- 2. Mode share of all trips ----
    shares = pd.DataFrame(
        {
            "base_share_%": sb["trips"] / sb["trips"].sum() * 100,
            "scen_share_%": ss["trips"] / ss["trips"].sum() * 100,
        }
    )
    shares["change_pp"] = shares["scen_share_%"] - shares["base_share_%"]

    # ---- 3. Matched trips: effect on the same trips, by baseline category ----
    m = base.merge(scen, on="trip_id", how="left", suffixes=("_b", "_s"), indicator=True)
    m["unserved"] = m["_merge"] == "left_only"          # in baseline, missing in scenario
    m["mode_group_s"] = m["mode_group_s"].fillna("unserved")
    m["dt_min"] = (m["trav_time_s"] - m["trav_time_b"]) / 60  # NaN for unserved trips
    m["mode_changed"] = (m["mode_group_s"] != m["mode_group_b"]) & ~m["unserved"]

    by_base_category = m.groupby("mode_group_b").agg(
        baseline_trips=("trip_id", "size"),
        unserved_pct=("unserved", lambda s: s.mean() * 100),
        mode_changed_pct=("mode_changed", lambda s: s.mean() * 100),
        median_dt_min=("dt_min", "median"),
        p90_dt_min=("dt_min", lambda s: s.quantile(0.9)),
        p95_dt_min=("dt_min", lambda s: s.quantile(0.95)),
        mean_dt_min=("dt_min", "mean"),
    )
    overall_unserved_pct = m["unserved"].mean() * 100

    # ---- 4. Mode shift matrix (rows: baseline category, columns: scenario category, % of row) ----
    shift = pd.crosstab(m["mode_group_b"], m["mode_group_s"], normalize="index") * 100

    # ---- save ----
    per_category.to_csv(out_dir / f"{scen_name}_per_category_summary.csv")
    shares.to_csv(out_dir / f"{scen_name}_mode_shares.csv")
    by_base_category.to_csv(out_dir / f"{scen_name}_matched_by_baseline_category.csv")
    shift.to_csv(out_dir / f"{scen_name}_mode_shift_matrix.csv")

    if verbose:
        print("\nPer-category summary:\n", per_category.round(2).to_string())
        print("\nMode shares:\n", shares.round(2).to_string())
        print("\nMatched trips by baseline category:\n", by_base_category.round(2).to_string())
        print(f"\nOverall unserved trips: {overall_unserved_pct:.2f}%")
        print("\nMode shift matrix (% of baseline category):\n", shift.round(1).to_string())

    return {
        "per_category": per_category,
        "shares": shares,
        "by_base_category": by_base_category,
        "shift": shift,
        "unserved_pct": overall_unserved_pct,
    }


if __name__ == "__main__":
    analyse(BASE_PATH, SCEN_PATH)
