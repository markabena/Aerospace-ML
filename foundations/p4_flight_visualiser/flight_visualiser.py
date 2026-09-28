"""
Project 4: Flight Visualiser
Aerospace-ML -- Foundations track

Purpose
-------
Turn P3's cleaned telemetry and event summary into a single annotated
ascent profile: altitude, velocity, acceleration, and dynamic pressure
stacked over time, with the throttle-down window and MECO marked
directly on the chart.

Input files (both produced by P3 -- run that first):
    Falcon9_Ascent_Telemetry_Cleaned.csv
    Falcon9_Ascent_Summary.csv

This is the point of building P3 and P4 as separate scripts instead of
one big one: P3 owns cleaning and analysis, P4 only reads its output
and draws it. Change how max-Q is detected in P3, and P4 doesn't need
to know or care -- it just plots whatever P3 decided.

New Python concepts introduced in this build
----------------------------------------------
    - matplotlib.pyplot   : the standard plotting library
    - Figure vs Axes      : one Figure (the whole image) can hold several
                             Axes (individual subplots) sharing an x-axis
    - .plot() / .fill_between() : drawing a line, and shading a region
    - .axvline() / .axvspan()   : marking a single moment or a time window
    - .annotate()          : labelling a specific point on the chart
    - plt.savefig()         : writing the figure to an image file
"""

import matplotlib
matplotlib.use("Agg")  # write straight to a file, no display needed
import matplotlib.pyplot as plt
import pandas as pd

TELEMETRY_FILE = "Falcon9_Ascent_Telemetry_Cleaned.csv"
SUMMARY_FILE = "Falcon9_Ascent_Summary.csv"
OUTPUT_IMAGE = "Flight_Ascent_Profile.png"


def load_inputs():
    try:
        telemetry = pd.read_csv(TELEMETRY_FILE)
        summary = pd.read_csv(SUMMARY_FILE).iloc[0]  # one-row file -> one Series
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e.filename} not found -- run P3_Flight_Data_Analyser.py first, "
            f"it produces both files this script needs."
        )
    return telemetry, summary


def mark_events(ax, summary, label_y_frac=0.9):
    """
    Draw the throttle-down window and MECO on a given subplot.

    >>> CONCEPT: this function takes an Axes object (`ax`) as an
    argument and draws directly onto it. Any of the four subplots
    below can be passed in -- the function doesn't care which one.
    """
    # axvspan: a shaded vertical band between two x-values -- good for
    # marking a WINDOW of time (the throttle-down period).
    ax.axvspan(
        summary["throttle_down_start_s"], summary["throttle_down_end_s"],
        color="orange", alpha=0.15, label="Throttle-down"
    )

    # axvline: a single vertical line -- good for marking ONE moment
    # (max-Q, MECO) rather than a range.
    ax.axvline(summary["max_q_time_s"], color="crimson", linestyle="--", linewidth=1, label="Max-Q")
    ax.axvline(summary["meco_time_s"], color="black", linestyle="-", linewidth=1, label="MECO")


def build_ascent_profile(telemetry, summary):
    t = telemetry["time_s"]

    # >>> CONCEPT: plt.subplots(4, 1, sharex=True) creates a Figure
    # containing 4 stacked Axes that all share the same x-axis -- move
    # along time on one panel and every panel lines up underneath it.
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 11))
    fig.suptitle("Falcon 9 Block 5 -- Simulated First-Stage Ascent Profile", fontsize=13, fontweight="bold")

    # --- Panel 1: Altitude ---
    ax = axes[0]
    ax.plot(t, telemetry["altitude_km"], color="steelblue", linewidth=1.5)
    mark_events(ax, summary)
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- Panel 2: Velocity ---
    ax = axes[1]
    ax.plot(t, telemetry["velocity_ms"], color="seagreen", linewidth=1.5)
    mark_events(ax, summary)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity", loc="left", fontsize=10)

    # --- Panel 3: Acceleration -- raw vs smoothed, so the noise-reduction
    # from P3 is actually visible rather than just asserted ---
    ax = axes[2]
    ax.plot(t, telemetry["acceleration_g"], color="lightgray", linewidth=1, label="Raw (sensor)")
    ax.plot(t, telemetry["acceleration_g_smoothed"], color="darkorange", linewidth=1.5, label="Smoothed")
    mark_events(ax, summary)
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Acceleration -- raw vs smoothed", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- Panel 4: Dynamic pressure, with max-Q explicitly called out ---
    ax = axes[3]
    ax.plot(t, telemetry["dynamic_pressure_kpa"], color="firebrick", linewidth=1.5)
    ax.fill_between(t, telemetry["dynamic_pressure_kpa"], color="firebrick", alpha=0.08)
    mark_events(ax, summary)

    # >>> CONCEPT: .annotate() labels a specific (x, y) point with text
    # and an optional arrow -- more precise than just a vertical line
    # when you want a number attached to the moment, not just its time.
    ax.annotate(
        f"Max-Q: {summary['max_q_kpa']:.1f} kPa\nat T+{summary['max_q_time_s']:.0f}s",
        xy=(summary["max_q_time_s"], summary["max_q_kpa"]),
        xytext=(summary["max_q_time_s"] + 15, summary["max_q_kpa"] + 4),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
    )
    ax.set_ylabel("Dynamic pressure (kPa)")
    ax.set_xlabel("Time since liftoff (s)")
    ax.set_title("Dynamic pressure", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    telemetry, summary = load_inputs()
    print(f"Loaded {len(telemetry)} cleaned telemetry rows and the event summary.\n")

    fig = build_ascent_profile(telemetry, summary)
    fig.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Saved: {OUTPUT_IMAGE}")
    print(
        f"Events plotted -- throttle-down: T+{summary['throttle_down_start_s']:.0f}s to "
        f"T+{summary['throttle_down_end_s']:.0f}s | max-Q: T+{summary['max_q_time_s']:.0f}s | "
        f"MECO: T+{summary['meco_time_s']:.0f}s"
    )


if __name__ == "__main__":
    main()
