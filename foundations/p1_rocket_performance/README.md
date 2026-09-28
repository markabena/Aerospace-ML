# P1 · Rocket Performance Evaluator

A command-line pre-launch analysis tool. Enter a single-stage vehicle's mass, specific impulse, thrust and launch angle; it returns a full performance report and a GO / WARN / ABORT readiness check.

## What it computes

| Quantity | Model |
|---|---|
| Ideal Δv | Tsiolkovsky rocket equation, Δv = Isp · g₀ · ln(m₀ / m_f) |
| Exhaust velocity, mass flow, burn time | v_e = Isp · g₀, ṁ = T / v_e, t_b = m_prop / ṁ |
| Gravity loss and net Δv | Constant-angle gravity loss over the burn |
| Thrust-to-weight, peak g-load | At liftoff and at propellant exhaustion |
| Apogee estimate | Energy method, h = v² / 2g |

**Readiness checks:** TWR below 1.0 aborts; marginal TWR, low Isp, near-empty mass ratio, very short burns and loads above 10 g raise warnings.

**Assumptions:** drag-free, flat Earth, constant thrust, single stage. The report states these at the bottom so the numbers are never read as more than they are.

## Run

```bash
python rocket_eval.py
```

Standard library only.

## Concepts

Functions, constants, input validation loops, `math`, formatted output.
