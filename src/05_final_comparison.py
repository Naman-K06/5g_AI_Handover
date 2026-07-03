
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from feature_ablation import FeatureAblationStudy, plot_ablation_results
from delay_throughput_analyzer import HandoverDelayModel, ThroughputAnalyzer
from statistical_analyzer import StatisticalAnalyzer



os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from physics_engine import get_network_state
from config import (
    WINDOW_SIZE, TIME_STEP, HEX_FILE, MODEL_FILE,
    SAVE_PATH, NUM_UES,
    RLF_SINR_THRESHOLD, RLF_TIME_LIMIT,
    HOF_SINR_THRESHOLD,
    PING_PONG_THRESHOLD,
    L3_FILTER_COEFFICIENT,
)

# ==================== AI HANDOVER PARAMETERS ====================
AI_STABILITY_MARGIN = 5.0   # dB gain required to trigger an optimisation HO
AI_MIN_STAY_TIME    = 4.0   # Minimum seconds on a cell before switching
CRITICAL_SINR       = -8.0  # Safety HO threshold — fires before RLF floor (-12 dB)


# -------------------
# HELPERS
# -------------------

def load_baseline_from_csv():
    """Load per-UE KPIs from the CSV logs written by 03_run_multi_user.py."""
    ue_log_dir = os.path.join(SAVE_PATH, "ue_logs")
    if not os.path.exists(ue_log_dir):
        raise FileNotFoundError(f"UE log folder not found: {ue_log_dir}")

    results = []
    for filename in sorted(os.listdir(ue_log_dir)):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(ue_log_dir, filename))
        if df.empty:
            continue

        ue_id    = int(df["ue_id"].iloc[0])
        ho_count = int((df["serving"].diff() != 0).sum())
        speed    = float(df["speed_kmh"].iloc[0])

        results.append({
            "UE_ID":     ue_id + 1,
            "HO_Base":   ho_count,
            "Ping_Base": int(df["ping_pong"].max()),
            "HOF_Base":  int(df["hof"].max()),
            "RLF_Base":  int(df["rlf"].max()),
            "Speed":     speed,
        })
    return pd.DataFrame(results)


def evaluate_ai(route_id, model, bs_coords):
    """
    Re-simulate a route with the AI handover policy and return KPI counts.

    L3 filtering is applied to both RSRP and SINR before any threshold decision,
    matching the methodology in Script 03. All thresholds are evaluated on
    filtered signals, not raw channel state.
    """
    route_path = f"data/routes/route_{route_id}.json"
    if not os.path.exists(route_path):
        return None

    with open(route_path, "r") as f:
        route_data = json.load(f)

    pts      = np.array(route_data["route_points"])
    cum_dist = np.array(route_data["cum_dist"])

    # Derive per-step speed from route geometry
    step_dists  = np.diff(cum_dist, prepend=cum_dist[0])
    step_speeds = step_dists / TIME_STEP

    K             = L3_FILTER_COEFFICIENT
    filtered_rsrp = None
    filtered_sinr = None

    initial_state = get_network_state(pts[0], bs_coords,
                                      cell_loads=np.full(len(bs_coords) * 3, 0.5))
    prev_shadow   = initial_state["shadowing"]
    srv_ai        = int(np.argmax(initial_state["rsrp"]))
    last_ai       = srv_ai

    ho_ai         = 0
    ping_ai       = 0
    hof_ai        = 0
    rlf_ai        = 0
    rlf_timer_ai  = 0.0
    time_since_ho = 100.0
    feat_hist     = []

    for i, p in enumerate(pts):
        speed_mps = float(step_speeds[i])

        state       = get_network_state(p, bs_coords, prev_shadow,
                                        cell_loads=np.full(len(bs_coords) * 3, 0.5))
        prev_shadow = state["shadowing"]

        if filtered_rsrp is None:
            filtered_rsrp = state["rsrp"].copy()
            filtered_sinr = state["sinr"].copy()
        else:
            filtered_rsrp = (1 - K) * filtered_rsrp + K * state["rsrp"]
            filtered_sinr = (1 - K) * filtered_sinr + K * state["sinr"]

        step_feat = list(filtered_rsrp) + list(filtered_sinr) + [speed_mps]
        feat_hist.append(step_feat)

        if i >= WINDOW_SIZE:
            window    = np.array(feat_hist[-WINDOW_SIZE:]).flatten().reshape(1, -1)
            pred_cell = int(model.predict(window)[0])

            current_sinr = float(filtered_sinr[srv_ai])
            target_sinr  = float(filtered_sinr[pred_cell])
            gain         = target_sinr - current_sinr

            do_ho = False
            if current_sinr < CRITICAL_SINR and pred_cell != srv_ai:
                do_ho = True
            elif time_since_ho >= AI_MIN_STAY_TIME and gain > AI_STABILITY_MARGIN:
                if pred_cell != srv_ai:
                    do_ho = True

            if do_ho:
                if target_sinr < HOF_SINR_THRESHOLD:
                    hof_ai += 1
                else:
                    if pred_cell == last_ai and time_since_ho <= PING_PONG_THRESHOLD:
                        ping_ai += 1
                    last_ai       = srv_ai
                    srv_ai        = pred_cell
                    ho_ai        += 1
                    time_since_ho = 0.0

        serving_sinr = float(filtered_sinr[srv_ai])
        if serving_sinr < RLF_SINR_THRESHOLD:
            rlf_timer_ai += TIME_STEP
            if rlf_timer_ai >= RLF_TIME_LIMIT:
                rlf_ai       += 1
                rlf_timer_ai  = 0.0
                srv_ai        = int(np.argmax(filtered_rsrp))
                last_ai       = srv_ai
                time_since_ho = 0.0
        else:
            rlf_timer_ai = 0.0

        time_since_ho += TIME_STEP

    route_len_km = float(cum_dist[-1]) / 1000.0

    return {
        "UE_ID":         route_id,
        "HO_AI":         ho_ai,
        "Ping_AI":       ping_ai,
        "HOF_AI":        hof_ai,
        "RLF_AI":        rlf_ai,
        "Route_Len_km":  route_len_km,
    }


def evaluate_ai_detailed(route_id, model, bs_coords):
    """
    Same as evaluate_ai but also returns per-step SINR and throughput timeseries
    for SINR quality and throughput comparison plots.
    """
    route_path = f"data/routes/route_{route_id}.json"
    if not os.path.exists(route_path):
        return None

    with open(route_path, "r") as f:
        route_data = json.load(f)

    pts      = np.array(route_data["route_points"])
    cum_dist = np.array(route_data["cum_dist"])
    step_dists  = np.diff(cum_dist, prepend=cum_dist[0])
    step_speeds = step_dists / TIME_STEP

    K             = L3_FILTER_COEFFICIENT
    filtered_rsrp = None
    filtered_sinr = None

    initial_state = get_network_state(pts[0], bs_coords,
                                      cell_loads=np.full(len(bs_coords) * 3, 0.5))
    prev_shadow   = initial_state["shadowing"]
    srv_ai        = int(np.argmax(initial_state["rsrp"]))
    last_ai       = srv_ai
    ho_ai = ping_ai = hof_ai = rlf_ai = 0
    rlf_timer_ai  = 0.0
    time_since_ho = 100.0
    feat_hist     = []

    sinr_trace = []
    tput_trace = []
    ho_events  = []

    for i, p in enumerate(pts):
        speed_mps = float(step_speeds[i])
        state       = get_network_state(p, bs_coords, prev_shadow,
                                        cell_loads=np.full(len(bs_coords) * 3, 0.5))
        prev_shadow = state["shadowing"]

        if filtered_rsrp is None:
            filtered_rsrp = state["rsrp"].copy()
            filtered_sinr = state["sinr"].copy()
        else:
            filtered_rsrp = (1 - K) * filtered_rsrp + K * state["rsrp"]
            filtered_sinr = (1 - K) * filtered_sinr + K * state["sinr"]

        step_feat = list(filtered_rsrp) + list(filtered_sinr) + [speed_mps]
        feat_hist.append(step_feat)

        if i >= WINDOW_SIZE:
            window    = np.array(feat_hist[-WINDOW_SIZE:]).flatten().reshape(1, -1)
            pred_cell = int(model.predict(window)[0])
            current_sinr = float(filtered_sinr[srv_ai])
            target_sinr  = float(filtered_sinr[pred_cell])
            gain         = target_sinr - current_sinr

            do_ho = False
            if current_sinr < CRITICAL_SINR and pred_cell != srv_ai:
                do_ho = True
            elif time_since_ho >= AI_MIN_STAY_TIME and gain > AI_STABILITY_MARGIN:
                if pred_cell != srv_ai:
                    do_ho = True

            if do_ho:
                if target_sinr < HOF_SINR_THRESHOLD:
                    hof_ai += 1
                else:
                    if pred_cell == last_ai and time_since_ho <= PING_PONG_THRESHOLD:
                        ping_ai += 1
                    last_ai       = srv_ai
                    srv_ai        = pred_cell
                    ho_ai        += 1
                    time_since_ho = 0.0
                    ho_events.append(i)

        serving_sinr = float(filtered_sinr[srv_ai])
        serving_tput = float(state["throughput"][srv_ai]) / 1e6   # Mbps

        if serving_sinr < RLF_SINR_THRESHOLD:
            rlf_timer_ai += TIME_STEP
            if rlf_timer_ai >= RLF_TIME_LIMIT:
                rlf_ai       += 1
                rlf_timer_ai  = 0.0
                srv_ai        = int(np.argmax(filtered_rsrp))
                last_ai       = srv_ai
                time_since_ho = 0.0
        else:
            rlf_timer_ai = 0.0

        time_since_ho += TIME_STEP
        sinr_trace.append(serving_sinr)
        tput_trace.append(serving_tput)

    route_len_km = float(cum_dist[-1]) / 1000.0
    return {
        "UE_ID":        route_id,
        "HO_AI":        ho_ai,
        "Ping_AI":      ping_ai,
        "HOF_AI":       hof_ai,
        "RLF_AI":       rlf_ai,
        "Route_Len_km": route_len_km,
        "sinr_trace":   sinr_trace,
        "tput_trace":   tput_trace,
        "ho_events":    ho_events,
    }


def evaluate_baseline_sinr(route_id, bs_coords):
    """
    Re-simulate the baseline (greedy best-RSRP) policy on a route and
    collect per-step SINR and throughput for comparison plots.
    """
    route_path = f"data/routes/route_{route_id}.json"
    if not os.path.exists(route_path):
        return None

    with open(route_path, "r") as f:
        route_data = json.load(f)

    pts      = np.array(route_data["route_points"])
    cum_dist = np.array(route_data["cum_dist"])

    K             = L3_FILTER_COEFFICIENT
    filtered_rsrp = None
    filtered_sinr = None
    prev_shadow   = None
    srv            = None

    sinr_trace = []
    tput_trace = []
    ho_events  = []

    for i, p in enumerate(pts):
        state       = get_network_state(p, bs_coords, prev_shadow,
                                        cell_loads=np.full(len(bs_coords) * 3, 0.5))
        prev_shadow = state["shadowing"]

        if filtered_rsrp is None:
            filtered_rsrp = state["rsrp"].copy()
            filtered_sinr = state["sinr"].copy()
            srv           = int(np.argmax(filtered_rsrp))
        else:
            filtered_rsrp = (1 - K) * filtered_rsrp + K * state["rsrp"]
            filtered_sinr = (1 - K) * filtered_sinr + K * state["sinr"]

        # 3GPP baseline: switch to best RSRP + hysteresis
        best = int(np.argmax(filtered_rsrp))
        if best != srv:
            ho_events.append(i)
            srv = best

        sinr_trace.append(float(filtered_sinr[srv]))
        tput_trace.append(float(state["throughput"][srv]) / 1e6)

    return {"sinr_trace": sinr_trace, "tput_trace": tput_trace, "ho_events": ho_events}


def reliability_score(hof, rlf, total_ho):
    """Composite reliability: fraction of events that were NOT failures."""
    total_events = total_ho + hof + rlf
    if total_events == 0:
        return 1.0
    return 1.0 - (hof + rlf) / total_events


def mean_ci(data, confidence=0.95):
    """Compute mean and 95% confidence interval half-width."""
    n    = len(data)
    mean = np.mean(data)
    se   = stats.sem(data)
    h    = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return round(float(mean), 3), round(float(h), 3)


def plot_cdf(ax, base_vals, ai_vals, title, xlabel):
    """Plot empirical CDF for baseline vs AI on a given axis."""
    for vals, label, color in [
        (base_vals, "3GPP Baseline", "#1f77b4"),
        (ai_vals,   "AI Optimised",  "#ff7f0e"),
    ]:
        sorted_v = np.sort(vals)
        cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.step(sorted_v, cdf, label=label, color=color, linewidth=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("CDF", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)


# -------------------
# MAIN
# -------------------
if __name__ == "__main__":
    print("\n===== AI-Based 5G Handover Optimisation: Final Comparison =====\n")

    # Load model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}. Run 04_train_ai.py first.")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    # Load base station coords
    if not os.path.exists(HEX_FILE):
        raise FileNotFoundError(f"HEX file not found: {HEX_FILE}. Run 01_preprocess.py first.")
    bs_coords = pd.read_csv(HEX_FILE).values

    # Load baseline KPIs
    baseline_df = load_baseline_from_csv()
    print(f"Loaded baseline KPIs for {len(baseline_df)} UEs.")

    # Evaluate AI on each route
    print(f"Evaluating AI on {NUM_UES} routes...")
    ai_results = []
    for r_id in range(1, NUM_UES + 1):
        result = evaluate_ai(r_id, model, bs_coords)
        if result is not None:
            ai_results.append(result)
        if r_id % max(1, NUM_UES // 10) == 0:
            print(f"  Evaluated {r_id}/{NUM_UES} routes")

    ai_df    = pd.DataFrame(ai_results)
    final_df = pd.merge(baseline_df, ai_df, on="UE_ID", how="inner")

    # -------------------
    # NORMALISED HO RATE (per km)
    # -------------------
    if "Route_Len_km" not in final_df.columns:
        final_df["Route_Len_km"] = 1.0

    final_df["HO_Rate_Base"] = final_df["HO_Base"] / final_df["Route_Len_km"].clip(lower=0.1)
    final_df["HO_Rate_AI"]   = final_df["HO_AI"]   / final_df["Route_Len_km"].clip(lower=0.1)

    # -------------------
    # RELIABILITY SCORE
    # -------------------
    final_df["Reliability_Base"] = final_df.apply(
        lambda r: reliability_score(r["HOF_Base"], r["RLF_Base"], r["HO_Base"]), axis=1
    )
    final_df["Reliability_AI"] = final_df.apply(
        lambda r: reliability_score(r["HOF_AI"], r["RLF_AI"], r["HO_AI"]), axis=1
    )

    # -------------------
    # TOTALS TABLE
    # -------------------
    totals = final_df.sum(numeric_only=True)

    print("\n============== TOTAL COMPARISON ==============")
    print(f"{'Metric':<25} {'Baseline':>10} {'AI':>10} {'Change':>10}")
    print("-" * 57)

    for metric, base_col, ai_col in [
        ("Handovers",  "HO_Base",   "HO_AI"),
        ("Ping-Pong",  "Ping_Base", "Ping_AI"),
        ("HOF",        "HOF_Base",  "HOF_AI"),
        ("RLF Events", "RLF_Base",  "RLF_AI"),
    ]:
        b     = totals[base_col]
        a     = totals[ai_col]
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {metric:<23} {b:>10.0f} {a:>10.0f} {sign}{delta:>9.0f}")

    ho_reduction  = 100 * (totals["HO_Base"] - totals["HO_AI"]) / max(totals["HO_Base"], 1)
    avg_rel_base  = final_df["Reliability_Base"].mean()
    avg_rel_ai    = final_df["Reliability_AI"].mean()
    avg_rate_base = final_df["HO_Rate_Base"].mean()
    avg_rate_ai   = final_df["HO_Rate_AI"].mean()

    print(f"\n  HO Signalling Reduction  : {ho_reduction:.1f}%")
    print(f"  Avg HO Rate (Base)       : {avg_rate_base:.2f} HO/km")
    print(f"  Avg HO Rate (AI)         : {avg_rate_ai:.2f} HO/km")
    print(f"  Avg Reliability (Base)   : {avg_rel_base:.3f}")
    print(f"  Avg Reliability (AI)     : {avg_rel_ai:.3f}")
    print("=" * 57)

    # -------------------
    # SPEED-STRATIFIED BREAKDOWN
    # -------------------
    bins   = [0, 40, 70, 120]
    labels = ["Low (<40 km/h)", "Med (40-70 km/h)", "High (>70 km/h)"]
    final_df["Speed_Bin"] = pd.cut(final_df["Speed"], bins=bins, labels=labels)

    print("\n--- Handover Reduction by Speed Bin ---")
    print(f"  {'Bin':<20} {'N':>4} {'Base HO':>8} {'AI HO':>8} "
          f"{'Reduction':>10} {'Rel Base':>10} {'Rel AI':>8}")
    print("  " + "-" * 70)
    for grp, gdf in final_df.groupby("Speed_Bin", observed=True):
        b   = gdf["HO_Base"].sum()
        a   = gdf["HO_AI"].sum()
        red = 100 * (b - a) / max(b, 1)
        rb  = gdf["Reliability_Base"].mean()
        ra  = gdf["Reliability_AI"].mean()
        print(f"  {str(grp):<20} {len(gdf):>4} {b:>8.0f} {a:>8.0f} "
              f"{red:>9.1f}% {rb:>10.3f} {ra:>8.3f}")

    # -------------------
    # CONFIDENCE INTERVALS (95%) on per-UE KPIs
    # -------------------
    print("\n--- 95% Confidence Intervals on Per-UE KPIs ---")
    ci_mapping = [
        ("Handovers",  "HO_Base",   "HO_AI"),
        ("Ping-Pong",  "Ping_Base", "Ping_AI"),
        ("HOF",        "HOF_Base",  "HOF_AI"),
        ("RLF",        "RLF_Base",  "RLF_AI"),
    ]
    ci_rows = []
    for label, bc, ac in ci_mapping:
        b_mean, b_ci = mean_ci(final_df[bc].values)
        a_mean, a_ci = mean_ci(final_df[ac].values)
        ci_rows.append({
            "KPI":            label,
            "Baseline_Mean":  b_mean,
            "Baseline_95CI":  f"+/-{b_ci}",
            "AI_Mean":        a_mean,
            "AI_95CI":        f"+/-{a_ci}",
        })
        print(f"  {label:<12} Baseline: {b_mean:.3f} +/-{b_ci}  |  AI: {a_mean:.3f} +/-{a_ci}")

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(SAVE_PATH, "kpi_confidence_intervals.csv"), index=False)
    print(f"  Confidence intervals saved.")

    # -------------------
    # PLOT 1 — BAR CHART
    # -------------------
    metrics   = ["Handovers", "Ping-Pong", "HOF", "RLF"]
    base_vals = [totals["HO_Base"], totals["Ping_Base"], totals["HOF_Base"], totals["RLF_Base"]]
    ai_vals_t = [totals["HO_AI"],   totals["Ping_AI"],   totals["HOF_AI"],   totals["RLF_AI"]]

    x = np.arange(len(metrics))
    fig_bar, ax_bar = plt.subplots(figsize=(11, 6))
    bars_b = ax_bar.bar(x - 0.2, base_vals, 0.4, label="3GPP Baseline", color="#1f77b4")
    bars_a = ax_bar.bar(x + 0.2, ai_vals_t, 0.4, label="AI Optimised",  color="#ff7f0e")
    for bar in list(bars_b) + list(bars_a):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics)
    ax_bar.set_ylabel("Cumulative Count (All UEs)")
    ax_bar.set_title(
        f"3GPP Baseline vs AI Optimised  "
        f"(Margin: {AI_STABILITY_MARGIN} dB | Min Stay: {AI_MIN_STAY_TIME} s)"
    )
    ax_bar.legend()
    ax_bar.grid(axis="y", linestyle="--", alpha=0.6)
    fig_bar.tight_layout()
    fig_bar.savefig(os.path.join(SAVE_PATH, "ho_comparison_optimised.png"), dpi=300, bbox_inches="tight")
    print(f"\nBar chart saved.")
    plt.close(fig_bar)

    # -------------------
    # PLOT 2 — CDF PLOTS (4-panel)
    # -------------------
    fig_cdf, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig_cdf.suptitle("CDF of Per-UE KPIs: 3GPP Baseline vs AI Optimised", fontsize=13)

    cdf_specs = [
        (axes[0, 0], final_df["HO_Base"],   final_df["HO_AI"],   "Handovers CDF",  "HO Count per UE"),
        (axes[0, 1], final_df["Ping_Base"],  final_df["Ping_AI"], "Ping-Pong CDF",  "Ping-Pong Count"),
        (axes[1, 0], final_df["HOF_Base"],   final_df["HOF_AI"],  "HOF CDF",        "HOF Count per UE"),
        (axes[1, 1], final_df["RLF_Base"],   final_df["RLF_AI"],  "RLF CDF",        "RLF Count per UE"),
    ]
    for ax, bv, av, title, xlabel in cdf_specs:
        plot_cdf(ax, bv.values, av.values, title, xlabel)

    fig_cdf.tight_layout()
    fig_cdf.savefig(os.path.join(SAVE_PATH, "cdf_comparison.png"), dpi=300, bbox_inches="tight")
    print("CDF plot saved.")
    plt.close(fig_cdf)

    # -------------------
    # PLOT 3 — BOX PLOTS
    # -------------------
    fig_box, axes_box = plt.subplots(1, 4, figsize=(16, 6))
    fig_box.suptitle("Per-UE KPI Distribution: 3GPP Baseline vs AI Optimised", fontsize=13)

    box_specs = [
        (axes_box[0], "HO_Base",   "HO_AI",   "Handovers"),
        (axes_box[1], "Ping_Base", "Ping_AI",  "Ping-Pong"),
        (axes_box[2], "HOF_Base",  "HOF_AI",   "HOF"),
        (axes_box[3], "RLF_Base",  "RLF_AI",   "RLF Events"),
    ]
    for ax, bc, ac, title in box_specs:
        bp = ax.boxplot(
            [final_df[bc].values, final_df[ac].values],
            labels=["Baseline", "AI"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#1f77b4")
        bp["boxes"][1].set_facecolor("#ff7f0e")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Count per UE", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig_box.tight_layout()
    fig_box.savefig(os.path.join(SAVE_PATH, "boxplot_comparison.png"), dpi=300, bbox_inches="tight")
    print("Box plot saved.")
    plt.close(fig_box)

    # -------------------
    # PLOT 4 — NORMALISED HO RATE by Speed Bin
    # -------------------
    speed_groups  = final_df.groupby("Speed_Bin", observed=True)
    bin_labels    = [str(b) for b in speed_groups.groups.keys()]
    rate_base_grp = [speed_groups.get_group(b)["HO_Rate_Base"].mean() for b in speed_groups.groups]
    rate_ai_grp   = [speed_groups.get_group(b)["HO_Rate_AI"].mean()   for b in speed_groups.groups]

    x_spd = np.arange(len(bin_labels))
    fig_spd, ax_spd = plt.subplots(figsize=(9, 5))
    ax_spd.bar(x_spd - 0.2, rate_base_grp, 0.4, label="3GPP Baseline", color="#1f77b4")
    ax_spd.bar(x_spd + 0.2, rate_ai_grp,   0.4, label="AI Optimised",  color="#ff7f0e")
    ax_spd.set_xticks(x_spd)
    ax_spd.set_xticklabels(bin_labels, fontsize=9)
    ax_spd.set_ylabel("HO Rate (handovers / km)")
    ax_spd.set_title("Normalised Handover Rate by UE Speed")
    ax_spd.legend()
    ax_spd.grid(axis="y", linestyle="--", alpha=0.6)
    fig_spd.tight_layout()
    fig_spd.savefig(os.path.join(SAVE_PATH, "ho_rate_by_speed.png"), dpi=300, bbox_inches="tight")
    print("HO rate by speed chart saved.")
    plt.close(fig_spd)

    # -------------------
    # PLOT 5 — RELIABILITY SCORE by Speed Bin
    # -------------------
    rel_base_grp = [speed_groups.get_group(b)["Reliability_Base"].mean() for b in speed_groups.groups]
    rel_ai_grp   = [speed_groups.get_group(b)["Reliability_AI"].mean()   for b in speed_groups.groups]

    fig_rel, ax_rel = plt.subplots(figsize=(9, 5))
    ax_rel.bar(x_spd - 0.2, rel_base_grp, 0.4, label="3GPP Baseline", color="#1f77b4")
    ax_rel.bar(x_spd + 0.2, rel_ai_grp,   0.4, label="AI Optimised",  color="#ff7f0e")
    ax_rel.set_xticks(x_spd)
    ax_rel.set_xticklabels(bin_labels, fontsize=9)
    ax_rel.set_ylabel("Reliability Score (0-1)")
    ax_rel.set_ylim(0, 1.05)
    ax_rel.set_title("Reliability Score by UE Speed Bin")
    ax_rel.legend()
    ax_rel.grid(axis="y", linestyle="--", alpha=0.6)
    fig_rel.tight_layout()
    fig_rel.savefig(os.path.join(SAVE_PATH, "reliability_by_speed.png"), dpi=300, bbox_inches="tight")
    print("Reliability by speed chart saved.")
    plt.close(fig_rel)

    # ===================================================================
    # NEW RESULT 1 — AI MODEL PERFORMANCE (Feature Importance)
    # ===================================================================
    print("\n[NEW] Generating AI Model Performance plots...")
    try:
        num_cells = len(bs_coords) * 3
        importances = model.feature_importances_

        rsrp_imp  = importances[:WINDOW_SIZE * num_cells].reshape(WINDOW_SIZE, num_cells).sum(axis=0)
        sinr_imp  = importances[WINDOW_SIZE * num_cells: WINDOW_SIZE * num_cells * 2].reshape(WINDOW_SIZE, num_cells).sum(axis=0)
        speed_imp = importances[WINDOW_SIZE * num_cells * 2:].sum()

        top_n = min(10, num_cells)
        top_rsrp_idx = np.argsort(rsrp_imp)[-top_n:][::-1]
        top_sinr_idx = np.argsort(sinr_imp)[-top_n:][::-1]

        fig_fi, axes_fi = plt.subplots(1, 3, figsize=(16, 5))
        fig_fi.suptitle("Random Forest Feature Importance Analysis", fontsize=13, fontweight="bold")

        axes_fi[0].barh([f"Cell {i} RSRP" for i in top_rsrp_idx],
                        rsrp_imp[top_rsrp_idx], color="#1f77b4")
        axes_fi[0].set_title(f"Top {top_n} RSRP Features")
        axes_fi[0].set_xlabel("Aggregated Importance")
        axes_fi[0].invert_yaxis()

        axes_fi[1].barh([f"Cell {i} SINR" for i in top_sinr_idx],
                        sinr_imp[top_sinr_idx], color="#ff7f0e")
        axes_fi[1].set_title(f"Top {top_n} SINR Features")
        axes_fi[1].set_xlabel("Aggregated Importance")
        axes_fi[1].invert_yaxis()

        type_labels = ["RSRP\n(all cells)", "SINR\n(all cells)", "Speed"]
        type_vals   = [rsrp_imp.sum(), sinr_imp.sum(), speed_imp]
        bars = axes_fi[2].bar(type_labels, type_vals,
                              color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        for bar in bars:
            axes_fi[2].text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{bar.get_height():.3f}", ha="center", fontsize=9)
        axes_fi[2].set_title("Feature Type Contribution")
        axes_fi[2].set_ylabel("Total Importance")

        fig_fi.tight_layout()
        fig_fi.savefig(os.path.join(SAVE_PATH, "feature_importance.png"), dpi=300, bbox_inches="tight")
        print(f"  Feature importance plot saved.")
        plt.close(fig_fi)
    except Exception as e:
        print(f"  [WARN] Feature importance plot skipped: {e}")

    # ===================================================================
    # NEW RESULT 2 — SINR QUALITY COMPARISON
    # ===================================================================
    print("\n[NEW] Computing per-UE mean SINR (re-simulating sample routes)...")
    try:
        SINR_SAMPLE = min(20, len(final_df))
        sample_ids  = final_df["UE_ID"].iloc[:SINR_SAMPLE].tolist()

        mean_sinr_base_list = []
        mean_sinr_ai_list   = []
        mean_tput_base_list = []
        mean_tput_ai_list   = []

        for rid in sample_ids:
            base_res = evaluate_baseline_sinr(rid, bs_coords)
            ai_res   = evaluate_ai_detailed(rid, model, bs_coords)
            if base_res and ai_res:
                mean_sinr_base_list.append(np.mean(base_res["sinr_trace"]))
                mean_sinr_ai_list.append(np.mean(ai_res["sinr_trace"]))
                mean_tput_base_list.append(np.mean(base_res["tput_trace"]))
                mean_tput_ai_list.append(np.mean(ai_res["tput_trace"]))

        if mean_sinr_base_list:
            fig_sq, axes_sq = plt.subplots(1, 2, figsize=(13, 5))
            fig_sq.suptitle("SINR Quality & Throughput: 3GPP Baseline vs AI Optimised",
                            fontsize=12, fontweight="bold")

            axes_sq[0].hist(mean_sinr_base_list, bins=10, alpha=0.7,
                            label="3GPP Baseline", color="#1f77b4", edgecolor="white")
            axes_sq[0].hist(mean_sinr_ai_list, bins=10, alpha=0.7,
                            label="AI Optimised",  color="#ff7f0e", edgecolor="white")
            axes_sq[0].axvline(np.mean(mean_sinr_base_list), color="#1f77b4",
                               linestyle="--", linewidth=2,
                               label=f"Base mu={np.mean(mean_sinr_base_list):.1f} dB")
            axes_sq[0].axvline(np.mean(mean_sinr_ai_list), color="#ff7f0e",
                               linestyle="--", linewidth=2,
                               label=f"AI mu={np.mean(mean_sinr_ai_list):.1f} dB")
            axes_sq[0].set_xlabel("Mean SINR per UE (dB)")
            axes_sq[0].set_ylabel("Count")
            axes_sq[0].set_title("SINR Distribution per UE")
            axes_sq[0].legend(fontsize=8)
            axes_sq[0].grid(True, linestyle="--", alpha=0.4)

            axes_sq[1].hist(mean_tput_base_list, bins=10, alpha=0.7,
                            label="3GPP Baseline", color="#1f77b4", edgecolor="white")
            axes_sq[1].hist(mean_tput_ai_list, bins=10, alpha=0.7,
                            label="AI Optimised",  color="#ff7f0e", edgecolor="white")
            axes_sq[1].axvline(np.mean(mean_tput_base_list), color="#1f77b4",
                               linestyle="--", linewidth=2,
                               label=f"Base mu={np.mean(mean_tput_base_list):.1f} Mbps")
            axes_sq[1].axvline(np.mean(mean_tput_ai_list), color="#ff7f0e",
                               linestyle="--", linewidth=2,
                               label=f"AI mu={np.mean(mean_tput_ai_list):.1f} Mbps")
            axes_sq[1].set_xlabel("Mean Throughput per UE (Mbps)")
            axes_sq[1].set_ylabel("Count")
            axes_sq[1].set_title("Throughput Distribution per UE")
            axes_sq[1].legend(fontsize=8)
            axes_sq[1].grid(True, linestyle="--", alpha=0.4)

            fig_sq.tight_layout()
            fig_sq.savefig(os.path.join(SAVE_PATH, "sinr_throughput_comparison.png"),
                           dpi=300, bbox_inches="tight")
            print(f"  SINR/Throughput comparison saved.")
            plt.close(fig_sq)
        else:
            print("  [WARN] No SINR data collected.")
    except Exception as e:
        print(f"  [WARN] SINR/Throughput plot skipped: {e}")

    # ===================================================================
    # NEW RESULT 3 — SINGLE UE TRAJECTORY PLOT
    # ===================================================================
    print("\n[NEW] Generating single-UE handover decision trajectory plot...")
    try:
        DEMO_ROUTE = int(final_df["UE_ID"].iloc[0])
        base_demo  = evaluate_baseline_sinr(DEMO_ROUTE, bs_coords)
        ai_demo    = evaluate_ai_detailed(DEMO_ROUTE, model, bs_coords)

        if base_demo and ai_demo:
            steps_b  = np.arange(len(base_demo["sinr_trace"])) * TIME_STEP
            steps_ai = np.arange(len(ai_demo["sinr_trace"]))   * TIME_STEP

            fig_traj, ax_traj = plt.subplots(figsize=(13, 5))
            ax_traj.plot(steps_b,  base_demo["sinr_trace"], color="#1f77b4",
                         linewidth=1.2, alpha=0.8, label="3GPP Baseline SINR")
            ax_traj.plot(steps_ai, ai_demo["sinr_trace"],   color="#ff7f0e",
                         linewidth=1.2, alpha=0.8, label="AI Optimised SINR")

            for ho_step in base_demo["ho_events"]:
                ax_traj.axvline(ho_step * TIME_STEP, color="#1f77b4",
                                linewidth=0.8, linestyle=":", alpha=0.5)
            for ho_step in ai_demo["ho_events"]:
                ax_traj.axvline(ho_step * TIME_STEP, color="#ff7f0e",
                                linewidth=0.8, linestyle=":", alpha=0.5)

            ax_traj.axhline(HOF_SINR_THRESHOLD, color="red",     linestyle="--",
                            linewidth=1.5, label=f"HOF Threshold ({HOF_SINR_THRESHOLD} dB)")
            ax_traj.axhline(RLF_SINR_THRESHOLD, color="darkred", linestyle="-.",
                            linewidth=1.5, label=f"RLF Threshold ({RLF_SINR_THRESHOLD} dB)")

            ax_traj.set_xlabel("Time (seconds)")
            ax_traj.set_ylabel("Serving Cell SINR (dB)")
            ax_traj.set_title(
                f"UE {DEMO_ROUTE} - SINR Trace & Handover Events\n"
                f"Baseline HOs: {len(base_demo['ho_events'])}  |  "
                f"AI HOs: {len(ai_demo['ho_events'])}"
            )
            ax_traj.legend(fontsize=8, loc="upper right")
            ax_traj.grid(True, linestyle="--", alpha=0.4)
            fig_traj.tight_layout()
            fig_traj.savefig(os.path.join(SAVE_PATH, "sinr_trajectory_demo.png"),
                             dpi=300, bbox_inches="tight")
            print(f"  Trajectory plot saved.")
            plt.close(fig_traj)
    except Exception as e:
        print(f"  [WARN] Trajectory plot skipped: {e}")

    # ===================================================================
    # NEW RESULT 4 — STATISTICAL SIGNIFICANCE TABLE
    # ===================================================================
    print("\n[NEW] Running statistical significance tests (paired t-test)...")
    try:
        sig_metrics = [
            ("Handovers", "HO_Base",   "HO_AI"),
            ("Ping-Pong", "Ping_Base", "Ping_AI"),
            ("HOF",       "HOF_Base",  "HOF_AI"),
            ("RLF",       "RLF_Base",  "RLF_AI"),
        ]

        print(f"\n  {'Metric':<15} {'Mean Base':>10} {'Mean AI':>10} "
              f"{'Delta%':>8} {'p-value':>10} {'Sig?':>6}")
        print("  " + "-" * 63)

        stat_rows = []
        for label, bc, ac in sig_metrics:
            b_vals = final_df[bc].values.astype(float)
            a_vals = final_df[ac].values.astype(float)
            t_stat, p_val = stats.ttest_rel(b_vals, a_vals)
            mean_b  = b_vals.mean()
            mean_a  = a_vals.mean()
            delta   = 100 * (mean_a - mean_b) / max(mean_b, 1e-9)
            sig_str = "Yes" if p_val < 0.05 else "No"
            print(f"  {label:<15} {mean_b:>10.2f} {mean_a:>10.2f} "
                  f"{delta:>7.1f}% {p_val:>10.5f} {sig_str:>6}")
            stat_rows.append({
                "Metric":        label,
                "Mean_Baseline": round(mean_b, 3),
                "Mean_AI":       round(mean_a, 3),
                "Change_pct":    round(delta, 1),
                "p_value":       round(p_val, 5),
                "Significant":   sig_str,
            })

        stats_df = pd.DataFrame(stat_rows)
        stats_df.to_csv(os.path.join(SAVE_PATH, "statistical_tests.csv"), index=False)
        print(f"\n  Statistical tests saved.")
    except Exception as e:
        print(f"  [WARN] Statistical tests skipped: {e}")

    # ===================================================================
    # NEW RESULT 5 — MASTER KPI SUMMARY TABLE
    # ===================================================================
    print("\n[NEW] Generating master KPI summary table...")
    try:
        totals_for_table = final_df.sum(numeric_only=True)
        ho_red  = 100 * (totals_for_table["HO_Base"]   - totals_for_table["HO_AI"])   / max(totals_for_table["HO_Base"],   1)
        pp_red  = 100 * (totals_for_table["Ping_Base"]  - totals_for_table["Ping_AI"]) / max(totals_for_table["Ping_Base"],  1)
        hof_chg = 100 * (totals_for_table["HOF_AI"]    - totals_for_table["HOF_Base"]) / max(totals_for_table["HOF_Base"],  1)
        rlf_chg = 100 * (totals_for_table["RLF_AI"]    - totals_for_table["RLF_Base"]) / max(totals_for_table["RLF_Base"],  1)

        summary_data = {
            "KPI": [
                "Total Handovers", "Ping-Pong HOs", "HO Failures (HOF)",
                "Radio Link Failures", "Avg Reliability Score", "Avg HO Rate (HO/km)"
            ],
            "3GPP_Baseline": [
                int(totals_for_table["HO_Base"]),
                int(totals_for_table["Ping_Base"]),
                int(totals_for_table["HOF_Base"]),
                int(totals_for_table["RLF_Base"]),
                f"{avg_rel_base:.3f}",
                f"{final_df['HO_Rate_Base'].mean():.2f}",
            ],
            "AI_Optimised": [
                int(totals_for_table["HO_AI"]),
                int(totals_for_table["Ping_AI"]),
                int(totals_for_table["HOF_AI"]),
                int(totals_for_table["RLF_AI"]),
                f"{avg_rel_ai:.3f}",
                f"{final_df['HO_Rate_AI'].mean():.2f}",
            ],
            "Change_%": [
                f"{-ho_red:+.1f}%",
                f"{-pp_red:+.1f}%",
                f"{+hof_chg:+.1f}%",
                f"{+rlf_chg:+.1f}%",
                f"{100*(avg_rel_ai - avg_rel_base)/max(avg_rel_base, 1e-9):+.1f}%",
                "--",
            ],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(SAVE_PATH, "kpi_summary_table.csv"), index=False)
        print(f"  Master KPI table saved.")
    except Exception as e:
        print(f"  [WARN] KPI summary table skipped: {e}")


    # ===================================================================
    # 06 — FEATURE ABLATION STUDY
    # ===================================================================
    print("\n[06] Running Feature Ablation Study...")
    try:
        from config import (AI_DATA_DIR, NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY,
                             WINDOW_SIZE as WS, N_ESTIMATORS, MAX_DEPTH, ROUTE_DIR,
                             SPEED_MIN, SPEED_MAX, AREA_W, AREA_H)
        import glob

        bs_coords_abl = bs_coords
        num_cells_abl = len(bs_coords_abl) * 3

        route_files = sorted(glob.glob(os.path.join("data/routes", "route_*.json")))
        route_pts_abl = []
        for rf in route_files:
            with open(rf) as f:
                rd = json.load(f)
            pts = rd.get("route_points", [])
            if len(pts) >= STEPS_PER_TRAJECTORY + WINDOW_SIZE:
                route_pts_abl.append(np.array(pts))

        X_abl, y_abl = [], []
        for traj_idx in range(min(NUM_TRAJECTORIES, 200)):  # cap for speed
            route_pts = route_pts_abl[traj_idx % len(route_pts_abl)]
            start = np.random.randint(0, max(1, len(route_pts) - STEPS_PER_TRAJECTORY - WINDOW_SIZE))
            waypoints = route_pts[start: start + STEPS_PER_TRAJECTORY + WINDOW_SIZE]
            speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
            temp_shadow = None
            buf = []
            serving = None
            for step_i in range(len(waypoints)):
                state = get_network_state(waypoints[step_i], bs_coords_abl, temp_shadow,
                                          np.full(num_cells_abl, 0.5))
                temp_shadow = state["shadowing"]
                buf.append(list(state["rsrp"]) + list(state["sinr"]) + [speed])
                if serving is None:
                    serving = int(np.argmax(state["sinr"]))
                if step_i >= WINDOW_SIZE:
                    window = np.array(buf[-WINDOW_SIZE:]).flatten()
                    X_abl.append(window)
                    y_abl.append(serving)

        X_abl = np.array(X_abl)
        y_abl = np.array(y_abl)

        ablation_df, _ = FeatureAblationStudy.run_full_ablation_study(
            X_abl, y_abl, num_cells_abl, SAVE_PATH,
            n_estimators=100, max_depth=15
        )
        plot_ablation_results(ablation_df, SAVE_PATH)
        print("  Ablation study complete.")
    except Exception as e:
        print(f"  [WARN] Ablation study skipped: {e}")

    # ===================================================================
    # 07 — DELAY & THROUGHPUT ANALYSIS
    # ===================================================================
    print("\n[07] Running Delay & Throughput Analysis...")
    try:
        from config import WINDOW_SIZE as WS2, TIME_STEP as TS2

        delay_df = HandoverDelayModel.delay_impact_analysis(
            final_df,
            policy_baseline_ho_col='HO_Base',
            policy_ai_ho_col='HO_AI',
            window_size_ts=WS2,
            time_step_s=TS2,
            include_hof_rlf=True,
            hof_col='HOF_Base',
            rlf_col='RLF_Base'
        )
        delay_df.to_csv(os.path.join(SAVE_PATH, "delay_analysis.csv"), index=False)
        HandoverDelayModel.plot_delay_analysis(
            delay_df,
            save_path=os.path.join(SAVE_PATH, "delay_analysis.png")
        )

        # Throughput comparison using SINR traces already collected
        tput_rows = []
        for rid in final_df["UE_ID"].iloc[:min(50, len(final_df))].tolist():
            base_res = evaluate_baseline_sinr(rid, bs_coords)
            ai_res   = evaluate_ai_detailed(rid, model, bs_coords)
            if base_res and ai_res:
                _, _, avg_base = ThroughputAnalyzer.compute_throughput_trajectory(
                    np.array(base_res["tput_trace"]), time_step_s=TS2)
                _, _, avg_ai = ThroughputAnalyzer.compute_throughput_trajectory(
                    np.array(ai_res["tput_trace"]), time_step_s=TS2)
                sinr_base_mean = np.mean(base_res["sinr_trace"])
                sinr_ai_mean   = np.mean(ai_res["sinr_trace"])
                tput_rows.append({
                    "Route_ID": rid,
                    "Throughput_Baseline_Mbps": avg_base,
                    "Throughput_AI_Mbps": avg_ai,
                    "Throughput_Improvement_Percent": (avg_ai - avg_base) / max(avg_base, 1e-9) * 100,
                    "SINR_Mean_Baseline_dB": sinr_base_mean,
                    "SINR_Mean_AI_dB": sinr_ai_mean,
                    "SINR_Improvement_dB": sinr_ai_mean - sinr_base_mean,
                })

        tput_df = pd.DataFrame(tput_rows)
        tput_df.to_csv(os.path.join(SAVE_PATH, "throughput_analysis.csv"), index=False)
        ThroughputAnalyzer.plot_throughput_comparison(
            tput_df,
            save_path=os.path.join(SAVE_PATH, "throughput_analysis.png")
        )
        print("  Delay & throughput analysis complete.")
    except Exception as e:
        print(f"  [WARN] Delay/Throughput analysis skipped: {e}")

    # ===================================================================
    # 08 — ADVANCED STATISTICAL ANALYSIS
    # ===================================================================
    print("\n[08] Running Advanced Statistical Analysis...")
    try:
        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        kpis_to_test = [
            ("Handovers",  "HO_Base",   "HO_AI"),
            ("Ping-Pong",  "Ping_Base", "Ping_AI"),
            ("HOF",        "HOF_Base",  "HOF_AI"),
            ("RLF",        "RLF_Base",  "RLF_AI"),
        ]

        ttest_results, mw_results = {}, {}
        for label, bc, ac in kpis_to_test:
            b_vals = final_df[bc].values.astype(float)
            a_vals = final_df[ac].values.astype(float)
            ttest_results[label] = analyzer.independent_ttest_with_ci(b_vals, a_vals, label)
            mw_results[label]    = analyzer.mann_whitney_u_test(b_vals, a_vals, label)

        analyzer.plot_ci_comparison(
            ttest_results,
            save_path=os.path.join(SAVE_PATH, "statistical_ci_comparison.png")
        )

        summary_df = StatisticalAnalyzer.create_statistical_summary_table(
            ttest_results,
            save_path=os.path.join(SAVE_PATH, "advanced_statistical_summary.csv")
        )

        mw_df = pd.DataFrame([{
            "KPI": k, "U_Stat": v["u_statistic"], "p_value": v["p_value"],
            "Rank_Biserial_r": v["rank_biserial_r"], "Effect_Size": v["effect_size"],
            "Significance": v["significance"]
        } for k, v in mw_results.items()])
        mw_df.to_csv(os.path.join(SAVE_PATH, "mann_whitney_results.csv"), index=False)
        print("  Advanced statistical analysis complete.")
    except Exception as e:
        print(f"  [WARN] Advanced statistical analysis skipped: {e}")



    # -------------------
    # SAVE ENRICHED CSV
    # -------------------
    out_csv = os.path.join(SAVE_PATH, "final_comparison.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"\nEnriched per-UE CSV saved to: {out_csv}")

    # ===================================================================
    # PRIORITY 2a — SINR CDF CURVES
    # ===================================================================
    print("\n[P2] Generating SINR CDF curves (full UE population)...")
    try:
        SINR_FULL_SAMPLE = min(50, len(final_df))
        all_sinr_base, all_sinr_ai = [], []

        for rid in final_df["UE_ID"].iloc[:SINR_FULL_SAMPLE].tolist():
            base_res = evaluate_baseline_sinr(rid, bs_coords)
            ai_res   = evaluate_ai_detailed(rid, model, bs_coords)
            if base_res:
                all_sinr_base.extend(base_res["sinr_trace"])
            if ai_res:
                all_sinr_ai.extend(ai_res["sinr_trace"])

        if all_sinr_base and all_sinr_ai:
            fig_sinr_cdf, ax_sinr_cdf = plt.subplots(figsize=(10, 6))
            for vals, label, color in [
                (all_sinr_base, "3GPP Baseline", "#1f77b4"),
                (all_sinr_ai,   "AI Optimised",  "#ff7f0e"),
            ]:
                sorted_v = np.sort(vals)
                cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
                ax_sinr_cdf.plot(sorted_v, cdf, label=label, color=color, linewidth=2)

            ax_sinr_cdf.axvline(HOF_SINR_THRESHOLD, color="red",     linestyle="--",
                                linewidth=1.5, label=f"HOF Threshold ({HOF_SINR_THRESHOLD} dB)")
            ax_sinr_cdf.axvline(RLF_SINR_THRESHOLD, color="darkred", linestyle="-.",
                                linewidth=1.5, label=f"RLF Threshold ({RLF_SINR_THRESHOLD} dB)")
            ax_sinr_cdf.set_xlabel("Serving Cell SINR (dB)", fontsize=11)
            ax_sinr_cdf.set_ylabel("CDF", fontsize=11)
            ax_sinr_cdf.set_title(
                f"SINR CDF - All UEs (n={SINR_FULL_SAMPLE} routes)\n"
                f"3GPP Baseline vs AI Optimised Handover Policy", fontsize=12
            )
            ax_sinr_cdf.legend(fontsize=9)
            ax_sinr_cdf.grid(True, linestyle="--", alpha=0.5)
            fig_sinr_cdf.tight_layout()
            fig_sinr_cdf.savefig(os.path.join(SAVE_PATH, "sinr_cdf_full.png"),
                                 dpi=300, bbox_inches="tight")
            print(f"  SINR CDF saved.")
            plt.close(fig_sinr_cdf)
    except Exception as e:
        print(f"  [WARN] SINR CDF skipped: {e}")

    # ===================================================================
    # PRIORITY 2b — THROUGHPUT CDF
    # ===================================================================
    print("\n[P2] Generating Throughput CDF curves...")
    try:
        all_tput_base, all_tput_ai = [], []
        for rid in final_df["UE_ID"].iloc[:SINR_FULL_SAMPLE].tolist():
            base_res = evaluate_baseline_sinr(rid, bs_coords)
            ai_res   = evaluate_ai_detailed(rid, model, bs_coords)
            if base_res:
                all_tput_base.extend(base_res["tput_trace"])
            if ai_res:
                all_tput_ai.extend(ai_res["tput_trace"])

        if all_tput_base and all_tput_ai:
            fig_tput_cdf, ax_tput_cdf = plt.subplots(figsize=(10, 6))
            for vals, label, color in [
                (all_tput_base, "3GPP Baseline", "#1f77b4"),
                (all_tput_ai,   "AI Optimised",  "#ff7f0e"),
            ]:
                sorted_v = np.sort(vals)
                cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
                ax_tput_cdf.plot(sorted_v, cdf, label=label, color=color, linewidth=2)

            ax_tput_cdf.set_xlabel("Instantaneous Throughput (Mbps)", fontsize=11)
            ax_tput_cdf.set_ylabel("CDF", fontsize=11)
            ax_tput_cdf.set_title(
                f"Throughput CDF - All UEs (n={SINR_FULL_SAMPLE} routes)\n"
                f"3GPP Baseline vs AI Optimised Handover Policy", fontsize=12
            )
            ax_tput_cdf.legend(fontsize=9)
            ax_tput_cdf.grid(True, linestyle="--", alpha=0.5)
            fig_tput_cdf.tight_layout()
            fig_tput_cdf.savefig(os.path.join(SAVE_PATH, "throughput_cdf_full.png"),
                                 dpi=300, bbox_inches="tight")
            print(f"  Throughput CDF saved.")
            plt.close(fig_tput_cdf)
    except Exception as e:
        print(f"  [WARN] Throughput CDF skipped: {e}")

    # ===================================================================
    # PRIORITY 2c — HO RATE vs UE SPEED (4 fine-grained bins)
    # ===================================================================
    print("\n[P2] Generating HO rate vs UE speed plot (4-bin granularity)...")
    try:
        speed_bins_fine   = [20, 40, 60, 80, 100]
        speed_labels_fine = ["20-40", "40-60", "60-80", "80-100"]
        final_df["Speed_Bin_Fine"] = pd.cut(
            final_df["Speed"], bins=speed_bins_fine,
            labels=speed_labels_fine, right=True
        )

        fine_groups   = final_df.groupby("Speed_Bin_Fine", observed=True)
        fine_bin_lbls = [str(b) for b in fine_groups.groups.keys()]
        fine_base     = [fine_groups.get_group(b)["HO_Rate_Base"].mean() for b in fine_groups.groups]
        fine_ai       = [fine_groups.get_group(b)["HO_Rate_AI"].mean()   for b in fine_groups.groups]
        fine_n        = [len(fine_groups.get_group(b))                    for b in fine_groups.groups]

        x_fine = np.arange(len(fine_bin_lbls))
        fig_speed, ax_speed = plt.subplots(figsize=(10, 6))
        bars_base = ax_speed.bar(x_fine - 0.2, fine_base, 0.4, label="3GPP Baseline", color="#1f77b4")
        bars_ai   = ax_speed.bar(x_fine + 0.2, fine_ai,   0.4, label="AI Optimised",  color="#ff7f0e")
        for bar in list(bars_base) + list(bars_ai):
            ax_speed.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.01,
                          f"{bar.get_height():.2f}", ha="center", fontsize=8)
        ax_speed.set_xticks(x_fine)
        ax_speed.set_xticklabels(
            [f"{lbl} km/h\n(n={n})" for lbl, n in zip(fine_bin_lbls, fine_n)], fontsize=9
        )
        ax_speed.set_ylabel("Avg HO Rate (handovers / km)", fontsize=11)
        ax_speed.set_title(
            "Handover Rate vs UE Speed: 3GPP Baseline vs AI Optimised\n"
            "(4-bin granularity: 20, 40, 60, 80, 100 km/h)", fontsize=12
        )
        ax_speed.legend(fontsize=10)
        ax_speed.grid(axis="y", linestyle="--", alpha=0.5)
        fig_speed.tight_layout()
        fig_speed.savefig(os.path.join(SAVE_PATH, "ho_rate_vs_speed_fine.png"),
                          dpi=300, bbox_inches="tight")
        print(f"  HO rate vs speed (fine) saved.")
        plt.close(fig_speed)
    except Exception as e:
        print(f"  [WARN] HO rate vs speed plot skipped: {e}")

    # ===================================================================
    # PRIORITY 2d — CELL LOAD DISTRIBUTION
    # ===================================================================
    print("\n[P2] Generating cell load distribution histogram...")
    try:
        ue_log_dir = os.path.join(SAVE_PATH, "ue_logs")
        cell_visit_counts = {}
        for fname in sorted(os.listdir(ue_log_dir)):
            if not fname.endswith(".csv"):
                continue
            ue_log = pd.read_csv(os.path.join(ue_log_dir, fname))
            if "serving" not in ue_log.columns:
                continue
            for cell_id, count in ue_log["serving"].value_counts().items():
                cell_visit_counts[int(cell_id)] = cell_visit_counts.get(int(cell_id), 0) + count

        if cell_visit_counts:
            num_cells_total = len(bs_coords) * 3
            all_cell_loads  = np.array(
                [cell_visit_counts.get(c, 0) for c in range(num_cells_total)], dtype=float
            )
            from config import CELL_CAPACITY, SIM_DURATION
            utilisation = np.clip(all_cell_loads / max(CELL_CAPACITY * SIM_DURATION, 1), 0, 1)

            fig_load, ax_load = plt.subplots(figsize=(12, 5))
            ax_load.bar(np.arange(num_cells_total), utilisation,
                        color="#2ca02c", edgecolor="white", alpha=0.85)
            ax_load.axhline(np.mean(utilisation), color="red", linestyle="--", linewidth=1.5,
                            label=f"Mean utilisation: {np.mean(utilisation):.2f}")
            ax_load.set_xlabel("Cell ID", fontsize=11)
            ax_load.set_ylabel("Utilisation Fraction (0-1)", fontsize=11)
            ax_load.set_title(
                f"Cell Load Distribution - All {num_cells_total} Cells "
                f"({len(bs_coords)} Sites x 3 Sectors)", fontsize=12
            )
            ax_load.legend(fontsize=10)
            ax_load.set_xlim(-1, num_cells_total)
            ax_load.grid(axis="y", linestyle="--", alpha=0.4)
            fig_load.tight_layout()
            fig_load.savefig(os.path.join(SAVE_PATH, "cell_load_distribution.png"),
                             dpi=300, bbox_inches="tight")

            load_df = pd.DataFrame({
                "Cell_ID":     np.arange(num_cells_total),
                "Visit_Count": all_cell_loads.astype(int),
                "Utilisation": np.round(utilisation, 4),
            })
            load_df.to_csv(os.path.join(SAVE_PATH, "cell_load_distribution.csv"), index=False)
            print(f"  Cell load histogram saved.")
            plt.close(fig_load)
    except Exception as e:
        print(f"  [WARN] Cell load distribution skipped: {e}")

    # ===================================================================
    # PRIORITY 3a — CONFUSION MATRIX
    # ===================================================================
    print("\n[P3] Generating ML Confusion Matrix (sampled test set)...")
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        num_cells_ml = len(bs_coords) * 3
        X_eval, y_eval = [], []
        EVAL_ROUTES    = min(10, len(final_df))

        for rid in final_df["UE_ID"].iloc[:EVAL_ROUTES].tolist():
            rpath = f"data/routes/route_{rid}.json"
            if not os.path.exists(rpath):
                continue
            with open(rpath) as f:
                rd = json.load(f)
            pts_eval  = np.array(rd["route_points"])
            cum_eval  = np.array(rd["cum_dist"])
            step_spd  = np.diff(cum_eval, prepend=cum_eval[0]) / TIME_STEP

            K_e           = L3_FILTER_COEFFICIENT
            frsrp_e       = None
            fsinr_e       = None
            feat_buf_e    = []
            eval_shadow_e = None
            eval_loads_e  = np.full(num_cells_ml, 0.5)

            for i_e, p_e in enumerate(pts_eval):
                st_e          = get_network_state(p_e, bs_coords, eval_shadow_e, eval_loads_e)
                eval_shadow_e = st_e["shadowing"]
                if frsrp_e is None:
                    frsrp_e = st_e["rsrp"].copy()
                    fsinr_e = st_e["sinr"].copy()
                else:
                    frsrp_e = (1 - K_e) * frsrp_e + K_e * st_e["rsrp"]
                    fsinr_e = (1 - K_e) * fsinr_e + K_e * st_e["sinr"]

                feat_buf_e.append(list(frsrp_e) + list(fsinr_e) + [float(step_spd[i_e])])

                if i_e >= WINDOW_SIZE:
                    window_e = np.array(feat_buf_e[-WINDOW_SIZE:]).flatten()
                    X_eval.append(window_e)
                    y_eval.append(int(np.argmax(fsinr_e)))

        if len(X_eval) > 0:
            X_eval_arr = np.array(X_eval)
            y_eval_arr = np.array(y_eval)
            y_pred_cm  = model.predict(X_eval_arr)

            active_cells = np.union1d(np.unique(y_eval_arr), np.unique(y_pred_cm))
            cm_active    = confusion_matrix(y_eval_arr, y_pred_cm, labels=active_cells)

            fig_cm, ax_cm = plt.subplots(
                figsize=(max(8, len(active_cells) * 0.45),
                         max(8, len(active_cells) * 0.45))
            )
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_active,
                                          display_labels=active_cells)
            disp.plot(ax=ax_cm, colorbar=True, cmap="Blues", xticks_rotation=90)
            ax_cm.set_title(
                f"Confusion Matrix - AI Handover Model\n"
                f"Active Cells: {len(active_cells)} | Samples: {len(y_eval_arr)}", fontsize=11
            )
            fig_cm.tight_layout()
            fig_cm.savefig(os.path.join(SAVE_PATH, "confusion_matrix.png"),
                           dpi=200, bbox_inches="tight")
            cm_acc = np.sum(y_pred_cm == y_eval_arr) / len(y_eval_arr)
            print(f"  Confusion matrix saved. Eval accuracy: {cm_acc:.4f}")
            plt.close(fig_cm)
    except Exception as e:
        print(f"  [WARN] Confusion matrix skipped: {e}")

    # ===================================================================
    # PRIORITY 3b — CROSS-VALIDATION SCORES
    # ===================================================================
    print("\n[P3] Running 5-fold cross-validation on evaluation set...")
    try:
        from sklearn.model_selection import cross_val_score
        from config import CV_FOLDS

        if len(X_eval) >= CV_FOLDS * 2:
            cv_scores = cross_val_score(
                model, X_eval_arr, y_eval_arr,
                cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
            )
            print(f"  {CV_FOLDS}-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

            cv_df = pd.DataFrame({
                "Fold":     np.arange(1, CV_FOLDS + 1),
                "Accuracy": np.round(cv_scores, 5),
            })
            cv_df.loc[len(cv_df)] = [
                "Mean +/- Std",
                f"{cv_scores.mean():.5f} +/- {cv_scores.std():.5f}"
            ]
            cv_df.to_csv(os.path.join(SAVE_PATH, "cross_validation_scores.csv"), index=False)
            print(f"  CV scores saved.")
        else:
            print("  [WARN] Insufficient eval samples for cross-validation.")
    except Exception as e:
        print(f"  [WARN] Cross-validation skipped: {e}")

    # ===================================================================
    # PRIORITY 3c — LEARNING CURVE
    # ===================================================================
    print("\n[P3] Generating Learning Curve...")
    try:
        from sklearn.model_selection import learning_curve

        if len(X_eval) >= 20:
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X_eval_arr, y_eval_arr,
                cv=min(CV_FOLDS, len(X_eval) // 4),
                train_sizes=np.linspace(0.2, 1.0, 6),
                scoring="accuracy", n_jobs=-1,
                shuffle=True, random_state=42,
            )
            train_mean = train_scores.mean(axis=1)
            train_std  = train_scores.std(axis=1)
            test_mean  = test_scores.mean(axis=1)
            test_std   = test_scores.std(axis=1)

            fig_lc, ax_lc = plt.subplots(figsize=(9, 6))
            ax_lc.plot(train_sizes_abs, train_mean, "o-", color="#1f77b4",
                       linewidth=2, label="Training Accuracy")
            ax_lc.fill_between(train_sizes_abs,
                               train_mean - train_std, train_mean + train_std,
                               alpha=0.2, color="#1f77b4")
            ax_lc.plot(train_sizes_abs, test_mean, "s-", color="#ff7f0e",
                       linewidth=2, label="Cross-Validation Accuracy")
            ax_lc.fill_between(train_sizes_abs,
                               test_mean - test_std, test_mean + test_std,
                               alpha=0.2, color="#ff7f0e")
            ax_lc.set_xlabel("Number of Training Samples", fontsize=11)
            ax_lc.set_ylabel("Accuracy", fontsize=11)
            ax_lc.set_title(
                "Learning Curve - Random Forest AI Handover Model\n"
                "Train vs Cross-Validation Accuracy", fontsize=12
            )
            ax_lc.legend(fontsize=10)
            ax_lc.set_ylim(0, 1.05)
            ax_lc.grid(True, linestyle="--", alpha=0.5)
            fig_lc.tight_layout()
            fig_lc.savefig(os.path.join(SAVE_PATH, "learning_curve.png"),
                           dpi=300, bbox_inches="tight")
            print(f"  Learning curve saved.")
            plt.close(fig_lc)
        else:
            print("  [WARN] Insufficient samples for learning curve.")
    except Exception as e:
        print(f"  [WARN] Learning curve skipped: {e}")

    # ===================================================================
    # PRIORITY 4a — HOF TRIGGER SINR ANALYSIS
    # ===================================================================
    print("\n[P4] Analysing SINR at each HOF trigger event...")
    try:
        from config import HYSTERESIS_DB, TTT_LIMIT

        hof_sinr_at_trigger_base = []
        hof_sinr_at_trigger_ai   = []
        HOF_SAMPLE = min(30, len(final_df))

        for rid in final_df["UE_ID"].iloc[:HOF_SAMPLE].tolist():
            rpath = f"data/routes/route_{rid}.json"
            if not os.path.exists(rpath):
                continue
            with open(rpath) as f:
                rd = json.load(f)
            pts_h   = np.array(rd["route_points"])
            cum_h   = np.array(rd["cum_dist"])
            spd_h   = np.diff(cum_h, prepend=cum_h[0]) / TIME_STEP
            loads_h = np.full(len(bs_coords) * 3, 0.5)
            K_h     = L3_FILTER_COEFFICIENT

            # --- Baseline HOF ---
            frsrp_h = None; fsinr_h = None; prev_shadow_h = None
            srv_h = None; candidate_h = -1; ttt_h = 0.0

            for i_h, p_h in enumerate(pts_h):
                st_h          = get_network_state(p_h, bs_coords, prev_shadow_h, loads_h)
                prev_shadow_h = st_h["shadowing"]
                if frsrp_h is None:
                    frsrp_h = st_h["rsrp"].copy()
                    fsinr_h = st_h["sinr"].copy()
                    srv_h   = int(np.argmax(frsrp_h))
                else:
                    frsrp_h = (1 - K_h) * frsrp_h + K_h * st_h["rsrp"]
                    fsinr_h = (1 - K_h) * fsinr_h + K_h * st_h["sinr"]

                best_h = int(np.argmax(frsrp_h))
                if best_h != srv_h and frsrp_h[best_h] > frsrp_h[srv_h] + HYSTERESIS_DB:
                    if best_h == candidate_h:
                        ttt_h += TIME_STEP
                    else:
                        candidate_h = best_h
                        ttt_h       = TIME_STEP
                    if ttt_h >= TTT_LIMIT:
                        target_sinr_h = float(fsinr_h[candidate_h])
                        if target_sinr_h < HOF_SINR_THRESHOLD:
                            hof_sinr_at_trigger_base.append(target_sinr_h)
                        ttt_h, candidate_h = 0.0, -1
                else:
                    ttt_h, candidate_h = 0.0, -1

            # --- AI HOF ---
            frsrp_ai2 = None; fsinr_ai2 = None; prev_sh2 = None
            srv_ai2 = None; tsh2 = 100.0; feat_h2 = []

            for i_h2, p_h2 in enumerate(pts_h):
                st_h2    = get_network_state(p_h2, bs_coords, prev_sh2, loads_h)
                prev_sh2 = st_h2["shadowing"]
                if frsrp_ai2 is None:
                    frsrp_ai2 = st_h2["rsrp"].copy()
                    fsinr_ai2 = st_h2["sinr"].copy()
                    srv_ai2   = int(np.argmax(frsrp_ai2))
                else:
                    frsrp_ai2 = (1 - K_h) * frsrp_ai2 + K_h * st_h2["rsrp"]
                    fsinr_ai2 = (1 - K_h) * fsinr_ai2 + K_h * st_h2["sinr"]

                feat_h2.append(list(frsrp_ai2) + list(fsinr_ai2) + [float(spd_h[i_h2])])
                if i_h2 >= WINDOW_SIZE:
                    win_h2  = np.array(feat_h2[-WINDOW_SIZE:]).flatten().reshape(1, -1)
                    pred_h2 = int(model.predict(win_h2)[0])
                    cur_sinr2 = float(fsinr_ai2[srv_ai2])
                    tgt_sinr2 = float(fsinr_ai2[pred_h2])
                    gain2     = tgt_sinr2 - cur_sinr2

                    do_ho2 = False
                    if cur_sinr2 < CRITICAL_SINR and pred_h2 != srv_ai2:
                        do_ho2 = True
                    elif tsh2 >= AI_MIN_STAY_TIME and gain2 > AI_STABILITY_MARGIN and pred_h2 != srv_ai2:
                        do_ho2 = True

                    if do_ho2 and tgt_sinr2 < HOF_SINR_THRESHOLD:
                        hof_sinr_at_trigger_ai.append(tgt_sinr2)
                tsh2 += TIME_STEP

        fig_hof, ax_hof = plt.subplots(figsize=(10, 6))
        if hof_sinr_at_trigger_base:
            ax_hof.hist(hof_sinr_at_trigger_base, bins=15, alpha=0.7, color="#1f77b4",
                        edgecolor="white",
                        label=f"3GPP Baseline HOF triggers (n={len(hof_sinr_at_trigger_base)})")
        if hof_sinr_at_trigger_ai:
            ax_hof.hist(hof_sinr_at_trigger_ai, bins=15, alpha=0.7, color="#ff7f0e",
                        edgecolor="white",
                        label=f"AI Optimised HOF triggers (n={len(hof_sinr_at_trigger_ai)})")
        ax_hof.axvline(HOF_SINR_THRESHOLD, color="red", linestyle="--", linewidth=2,
                       label=f"HOF Threshold ({HOF_SINR_THRESHOLD} dB)")
        ax_hof.set_xlabel("Target Cell SINR at HOF Trigger Event (dB)", fontsize=11)
        ax_hof.set_ylabel("Event Count", fontsize=11)
        ax_hof.set_title(
            "HOF Trigger SINR Analysis\n"
            "Distribution of Serving-Cell SINR When Handover Failure Occurs", fontsize=12
        )
        ax_hof.legend(fontsize=9)
        ax_hof.grid(True, linestyle="--", alpha=0.5)
        fig_hof.tight_layout()
        fig_hof.savefig(os.path.join(SAVE_PATH, "hof_trigger_sinr_analysis.png"),
                        dpi=300, bbox_inches="tight")
        print(f"  HOF trigger SINR analysis saved.")
        plt.close(fig_hof)
    except Exception as e:
        print(f"  [WARN] HOF trigger analysis skipped: {e}")

    # ===================================================================
    # PRIORITY 4b — PING-PONG EVENT MAP
    # ===================================================================
    print("\n[P4] Generating Ping-Pong Event Map...")
    try:
        import json as _json_pp
        with open("data/map_metadata.json", "r") as f_pp:
            meta_pp = _json_pp.load(f_pp)

        pp_x_list, pp_y_list = [], []
        ue_log_dir_pp = os.path.join(SAVE_PATH, "ue_logs")
        for fname_pp in sorted(os.listdir(ue_log_dir_pp)):
            if not fname_pp.endswith(".csv"):
                continue
            ue_df_pp = pd.read_csv(os.path.join(ue_log_dir_pp, fname_pp))
            if "ping_pong" not in ue_df_pp.columns:
                continue
            pp_events = ue_df_pp[ue_df_pp["ping_pong"].diff() > 0]
            pp_x_list.extend(pp_events["x"].tolist())
            pp_y_list.extend(pp_events["y"].tolist())

        fig_pp, ax_pp = plt.subplots(figsize=(12, 12), facecolor="#0D1117")
        ax_pp.set_facecolor("#0D1117")

        import osmnx as ox
        G_pp = ox.load_graphml("data/road_network.graphml")
        _, edges_pp = ox.graph_to_gdfs(G_pp)
        sc_pp = meta_pp["scale_factor"]
        ox_pp = meta_pp["offset_x"]
        oy_pp = meta_pp["offset_y"]
        mx_pp = meta_pp["minx"]
        my_pp = meta_pp["miny"]

        for geom_pp in edges_pp.geometry:
            xs_pp, ys_pp = geom_pp.xy
            ax_pp.plot(
                (np.array(xs_pp) - mx_pp) * sc_pp + ox_pp,
                (np.array(ys_pp) - my_pp) * sc_pp + oy_pp,
                color="#1B2129", linewidth=0.7, alpha=0.5, zorder=1
            )

        if pp_x_list:
            ax_pp.scatter(pp_x_list, pp_y_list, c="yellow", s=20, alpha=0.6, zorder=5,
                          label=f"Ping-Pong Events (n={len(pp_x_list)})")
            ax_pp.legend(loc="upper right", facecolor="#111", edgecolor="#444",
                         labelcolor="white", fontsize=10)

        ax_pp.set_xlim(0, meta_pp["AREA_W"])
        ax_pp.set_ylim(0, meta_pp["AREA_H"])
        ax_pp.axis("off")
        ax_pp.set_title(
            f"Ping-Pong Event Map - {meta_pp.get('PLACE', 'Chandigarh')}\n"
            f"Total Events: {len(pp_x_list)}",
            color="white", fontsize=14, pad=15
        )
        fig_pp.tight_layout()
        fig_pp.savefig(os.path.join(SAVE_PATH, "ping_pong_event_map.png"),
                       dpi=300, bbox_inches="tight", facecolor="#0D1117")
        print(f"  Ping-pong event map saved. Events plotted: {len(pp_x_list)}")
        plt.close(fig_pp)
    except Exception as e:
        print(f"  [WARN] Ping-pong event map skipped: {e}")

    # ===================================================================
    # PRIORITY 4c — HANDOVER EVENT DENSITY HEATMAP
    # ===================================================================
    print("\n[P4] Generating Handover Event Density Heatmap...")
    try:
        import json as _json_ho
        with open("data/map_metadata.json", "r") as f_ho:
            meta_ho = _json_ho.load(f_ho)

        ho_x_list, ho_y_list = [], []
        ue_log_dir_ho = os.path.join(SAVE_PATH, "ue_logs")
        for fname_ho in sorted(os.listdir(ue_log_dir_ho)):
            if not fname_ho.endswith(".csv"):
                continue
            ue_df_ho = pd.read_csv(os.path.join(ue_log_dir_ho, fname_ho))
            if "serving" not in ue_df_ho.columns:
                continue
            ho_events_ho = ue_df_ho[ue_df_ho["serving"].diff() != 0].iloc[1:]
            ho_x_list.extend(ho_events_ho["x"].tolist())
            ho_y_list.extend(ho_events_ho["y"].tolist())

        if ho_x_list:
            fig_ho, ax_ho = plt.subplots(figsize=(12, 12), facecolor="#0D1117")
            ax_ho.set_facecolor("#0D1117")

            # Reuse edges_pp loaded in the ping-pong block above
            for geom_ho in edges_pp.geometry:
                xs_ho, ys_ho = geom_ho.xy
                ax_ho.plot(
                    (np.array(xs_ho) - mx_pp) * sc_pp + ox_pp,
                    (np.array(ys_ho) - my_pp) * sc_pp + oy_pp,
                    color="#1B2129", linewidth=0.7, alpha=0.5, zorder=1
                )

            AREA_W_ho = meta_ho["AREA_W"]
            AREA_H_ho = meta_ho["AREA_H"]
            h_ho, xedges, yedges = np.histogram2d(
                ho_x_list, ho_y_list, bins=80,
                range=[[0, AREA_W_ho], [0, AREA_H_ho]]
            )
            h_ho = h_ho.T
            im_ho = ax_ho.imshow(
                h_ho, origin="lower",
                extent=[0, AREA_W_ho, 0, AREA_H_ho],
                cmap="hot", alpha=0.75, zorder=3, aspect="auto"
            )
            plt.colorbar(im_ho, ax=ax_ho, label="Handover Event Count",
                         fraction=0.03, pad=0.02)
            ax_ho.set_xlim(0, AREA_W_ho)
            ax_ho.set_ylim(0, AREA_H_ho)
            ax_ho.axis("off")
            ax_ho.set_title(
                f"Handover Event Density Map - {meta_ho.get('PLACE', 'Chandigarh')}\n"
                f"Total Handover Events: {len(ho_x_list)}",
                color="white", fontsize=14, pad=15
            )
            fig_ho.tight_layout()
            fig_ho.savefig(os.path.join(SAVE_PATH, "handover_density_heatmap.png"),
                           dpi=300, bbox_inches="tight", facecolor="#0D1117")
            print(f"  Handover density heatmap saved. Events plotted: {len(ho_x_list)}")
            plt.close(fig_ho)
        else:
            print("  [WARN] No handover events found in UE logs for heatmap.")
    except Exception as e:
        print(f"  [WARN] Handover density heatmap skipped: {e}")

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON COMPLETE - All outputs saved to: {SAVE_PATH}")
    print(f"{'='*70}")
    print(f"  PRIORITY 1 - Core KPI Comparison:")
    print(f"    kpi_summary_table.csv         | ho_comparison_optimised.png")
    print(f"    final_comparison.csv          | statistical_tests.csv")
    print(f"    kpi_confidence_intervals.csv")
    print(f"  PRIORITY 2 - Statistical & Distribution Results:")
    print(f"    sinr_cdf_full.png             | throughput_cdf_full.png")
    print(f"    ho_rate_vs_speed_fine.png     | cell_load_distribution.png")
    print(f"    cell_load_distribution.csv    | cdf_comparison.png")
    print(f"    boxplot_comparison.png        | ho_rate_by_speed.png")
    print(f"    reliability_by_speed.png      | sinr_throughput_comparison.png")
    print(f"  PRIORITY 3 - Machine Learning Evaluation:")
    print(f"    feature_importance.png        | confusion_matrix.png")
    print(f"    learning_curve.png            | cross_validation_scores.csv")
    print(f"    sinr_trajectory_demo.png")
    print(f"  PRIORITY 4 - Scenario-Based Results:")
    print(f"    hof_trigger_sinr_analysis.png | ping_pong_event_map.png")
    print(f"    handover_density_heatmap.png")
    print(f"{'='*70}\n")