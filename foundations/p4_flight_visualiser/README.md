# P4 · Flight Visualiser

Turns P3's cleaned telemetry and event summary into one annotated ascent profile: altitude, velocity, acceleration and dynamic pressure stacked on a shared time axis, with max-Q, the throttle-down window and MECO marked on the chart.

![Annotated ascent profile](Flight_Ascent_Profile.png)

P3 owns cleaning and analysis; P4 only reads P3's output and draws it. If max-Q detection changes in P3, P4 needs no change.

## Run

Run P3 first, then:

```bash
python flight_visualiser.py
```

## Concepts

Figure vs Axes, shared-axis subplots, `fill_between`, `axvline` / `axvspan` event marking, `annotate`, `savefig`.
