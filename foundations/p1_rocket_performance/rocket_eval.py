#!/usr/bin/env python3
"""Rocket Performance Evaluator — pre-launch telemetry analysis tool."""

import math

G0 = 9.80665  # standard gravity, m/s²


def prompt_float(label, unit, min_exclusive=0, max_val=None):
    """Prompt until the user enters a valid float within bounds."""
    while True:
        try:
            val = float(input(f"  {label:<32} ({unit}) : "))
        except ValueError:
            print("    ! Enter a numeric value.")
            continue
        if val <= min_exclusive:
            print(f"    ! Must be > {min_exclusive}.")
            continue
        if max_val is not None and val > max_val:
            print(f"    ! Must be <= {max_val}.")
            continue
        return val


def rule(char="-", width=54):
    print(char * width)


def row(label, value, unit=""):
    suffix = f"  {unit}" if unit else ""
    print(f"  {label:<32} {value}{suffix}")


# ── Equations ────────────────────────────────────────────────────────────────

def tsiolkovsky(isp, wet_mass, dry_mass):
    """Ideal delta-v via Tsiolkovsky rocket equation (m/s)."""
    return isp * G0 * math.log(wet_mass / dry_mass)

def exhaust_velocity(isp):
    """Effective exhaust velocity (m/s)."""
    return isp * G0

def mass_flow_rate(thrust, ve):
    """Propellant mass flow rate (kg/s)."""
    return thrust / ve

def burn_time(prop_mass, mdot):
    """Constant-thrust burn duration (s)."""
    return prop_mass / mdot

def thrust_to_weight(thrust, wet_mass):
    """Liftoff thrust-to-weight ratio (dimensionless)."""
    return thrust / (wet_mass * G0)

def gravity_loss(angle_deg, t_burn):
    """Simplified gravity loss over burn (m/s). angle_deg from vertical."""
    return G0 * math.cos(math.radians(angle_deg)) * t_burn

def apogee_estimate(net_dv):
    """Drag-free apogee via energy method (m). h = v²/(2g)."""
    return (net_dv ** 2) / (2 * G0)

def max_gload(thrust, dry_mass):
    """Peak structural load at propellant exhaustion (g)."""
    return (thrust / dry_mass) / G0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rule("=")
    print("  ROCKET PERFORMANCE EVALUATOR")
    print("  Pre-Launch Telemetry Analysis")
    rule("=")

    # ── Inputs ────────────────────────────────────────────────────────────
    print()
    print("  VEHICLE PARAMETERS")
    rule()
    dry_mass      = prompt_float("Dry / structural mass",   "kg")
    prop_mass     = prompt_float("Propellant mass",          "kg")
    isp           = prompt_float("Specific impulse (Isp)",  "s")
    thrust        = prompt_float("Thrust at liftoff",        "N")

    print()
    print("  TRAJECTORY")
    rule()
    angle_deg = prompt_float("Launch angle from vertical", "deg",
                             min_exclusive=-1, max_val=90)

    # ── Derived quantities ────────────────────────────────────────────────
    wet_mass   = dry_mass + prop_mass
    mass_ratio = wet_mass / dry_mass

    ve         = exhaust_velocity(isp)
    mdot       = mass_flow_rate(thrust, ve)
    t_burn     = burn_time(prop_mass, mdot)

    ideal_dv   = tsiolkovsky(isp, wet_mass, dry_mass)
    grav_loss  = gravity_loss(angle_deg, t_burn)
    net_dv     = max(ideal_dv - grav_loss, 0.0)

    twr        = thrust_to_weight(thrust, wet_mass)
    apogee_m   = apogee_estimate(net_dv)
    peak_g     = max_gload(thrust, dry_mass)

    # ── Report ────────────────────────────────────────────────────────────
    print()
    rule("=")
    print("  TELEMETRY REPORT")
    rule("=")

    print()
    print("  VEHICLE")
    rule()
    row("Wet mass (at liftoff)",   f"{wet_mass:>12,.2f}", "kg")
    row("Dry mass (at burnout)",   f"{dry_mass:>12,.2f}", "kg")
    row("Propellant mass",         f"{prop_mass:>12,.2f}", "kg")
    row("Mass ratio  m0 / mf",      f"{mass_ratio:>12.4f}")

    print()
    print("  PROPULSION")
    rule()
    row("Specific impulse (Isp)",  f"{isp:>12.2f}", "s")
    row("Exhaust velocity (ve)",   f"{ve:>12.2f}", "m/s")
    row("Thrust",                  f"{thrust:>12,.2f}", "N")
    row("Mass flow rate",          f"{mdot:>12.4f}", "kg/s")
    row("Burn time",               f"{t_burn:>12.2f}", "s")

    print()
    print("  PERFORMANCE")
    rule()
    row("Ideal dv  (Tsiolkovsky)", f"{ideal_dv:>12.2f}", "m/s")
    row("Gravity loss (est.)",     f"{grav_loss:>12.2f}", "m/s")
    row("Net dv",                  f"{net_dv:>12.2f}", "m/s")
    row("Thrust-to-weight ratio",  f"{twr:>12.4f}")
    row("Peak structural load",    f"{peak_g:>12.2f}", "g")
    row("Apogee estimate",
        f"{apogee_m:>12,.0f}", f"m  ({apogee_m / 1000:.2f} km)")

    # ── Launch readiness ──────────────────────────────────────────────────
    print()
    print("  LAUNCH READINESS")
    rule()
    flags = []

    if twr < 1.0:
        flags.append(("ABORT", f"TWR {twr:.3f} < 1.0 - insufficient thrust to lift off."))
    elif twr < 1.3:
        flags.append(("WARN",  f"TWR {twr:.3f} - marginal liftoff margin (recommend >= 1.3)."))

    if isp < 200:
        flags.append(("WARN",  f"Isp {isp:.0f} s - below typical chemical minimum (~200 s)."))

    if mass_ratio < 1.05:
        flags.append(("WARN",  f"Mass ratio {mass_ratio:.3f} - nearly no propellant loaded."))

    if t_burn < 1.0:
        flags.append(("WARN",  f"Burn time {t_burn:.2f} s - extremely short; verify inputs."))

    if peak_g > 10:
        flags.append(("WARN",  f"Peak load {peak_g:.1f} g - verify structural limits."))

    if not flags:
        print("  [  GO  ]  All parameters within nominal range.")
    else:
        for tag, msg in flags:
            bracket = f"[{tag:^5}]"
            print(f"  {bracket}  {msg}")

    print()
    rule("=")
    print("  Model: drag-free, flat-Earth, constant thrust, single stage.")
    print("  Gravity loss assumes constant angle throughout burn.")
    rule("=")
    print()


if __name__ == "__main__":
    main()
