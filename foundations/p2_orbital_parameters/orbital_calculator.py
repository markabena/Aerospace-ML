"""
Project 2: Orbital Parameter Calculator
Aerospace-ML -- Foundations track

Purpose
-------
Given basic orbital inputs (a single altitude for a circular orbit, or a
periapsis + apoapsis pair for an elliptical one), compute the numbers that
describe the orbit: circular velocity, orbital period, semi-major axis,
eccentricity, and velocity at periapsis/apoapsis (vis-viva equation).

New Python concepts introduced in this build
----------------------------------------------
(marked inline with ">>> CONCEPT" comments where they're actually used)
    - Dictionaries : storing celestial body data as name -> constants
    - Tuples       : bundling related values (GM, radius) and returning
                     more than one result from a function
    - try/except   : catching bad user input without crashing the program

Validated against real orbits in run_self_tests(): the ISS (near-circular
LEO), geostationary orbit (GEO), and a geostationary transfer orbit (GTO).
"""

import math

# ---------------------------------------------------------------------------
# >>> CONCEPT: Dictionaries
# A dictionary maps a KEY (a body's name, as a string) to a VALUE.
# Here the value is itself a TUPLE: (GM in km^3/s^2, mean radius in km).
# Bundling GM and radius together in one tuple means you can never
# accidentally pair Earth's GM with Mars's radius by mistake -- they
# travel together as a single unit.
# ---------------------------------------------------------------------------
CELESTIAL_BODIES = {
    "Earth": (398_600.4418, 6378.137),
    "Moon": (4_902.8000, 1737.4),
    "Mars": (42_828.3, 3389.5),
}


class OrbitInputError(ValueError):
    """Raised when an orbital input fails a physical sanity check."""
    pass


def _get_body_constants(body):
    """
    Look up (GM, radius) for a body name and return the tuple.

    >>> CONCEPT: dict.get() with a default of None, then checking for
    None, is a clean way to validate a key exists before you use it --
    versus dict[key], which would crash with a KeyError on a typo.
    """
    body = body.strip().title()
    constants = CELESTIAL_BODIES.get(body)
    if constants is None:
        valid = ", ".join(CELESTIAL_BODIES.keys())
        raise OrbitInputError(f"Unknown body '{body}'. Choose from: {valid}")
    return constants  # caller unpacks this tuple: gm, radius = ...


def validate_radius(radius_km, body_radius_km, label="orbital radius"):
    """Sanity check: an orbit's radius must be above the body's surface."""
    if radius_km <= body_radius_km:
        raise OrbitInputError(
            f"{label} ({radius_km:.1f} km) must be greater than the body's "
            f"radius ({body_radius_km:.1f} km) -- the orbit would be "
            f"underground."
        )
    return radius_km


# ---------------------------------------------------------------------------
# CIRCULAR ORBIT FUNCTIONS
# ---------------------------------------------------------------------------
def circular_orbital_velocity(altitude_km, body="Earth"):
    """
    v = sqrt(GM / r)
    Speed needed to hold a perfectly circular orbit at a given altitude.
    """
    gm, body_radius = _get_body_constants(body)
    r = validate_radius(altitude_km + body_radius, body_radius)
    return math.sqrt(gm / r)


def orbital_period(altitude_km, body="Earth"):
    """
    T = 2*pi*sqrt(r^3 / GM)
    Time for one full circular orbit, in seconds.
    """
    gm, body_radius = _get_body_constants(body)
    r = validate_radius(altitude_km + body_radius, body_radius)
    return 2 * math.pi * math.sqrt(r ** 3 / gm)


# ---------------------------------------------------------------------------
# ELLIPTICAL ORBIT FUNCTIONS (apoapsis / periapsis)
# ---------------------------------------------------------------------------
def semi_major_axis(periapsis_alt_km, apoapsis_alt_km, body="Earth"):
    """
    a = (r_periapsis + r_apoapsis) / 2
    The semi-major axis is the 'average radius' of an ellipse -- half the
    sum of the closest and farthest points from the body's center.
    """
    _, body_radius = _get_body_constants(body)
    r_p = periapsis_alt_km + body_radius
    r_a = apoapsis_alt_km + body_radius
    if r_a < r_p:
        raise OrbitInputError(
            "Apoapsis must be >= periapsis (the farthest point can't be "
            "closer than the closest point)."
        )
    validate_radius(r_p, body_radius, "periapsis radius")
    return (r_p + r_a) / 2


def eccentricity(periapsis_alt_km, apoapsis_alt_km, body="Earth"):
    """
    e = (r_a - r_p) / (r_a + r_p)
    0 = perfect circle. Closer to 1 = a more stretched-out ellipse.
    """
    _, body_radius = _get_body_constants(body)
    r_p = periapsis_alt_km + body_radius
    r_a = apoapsis_alt_km + body_radius
    return (r_a - r_p) / (r_a + r_p)


def vis_viva_velocity(radius_km, semi_major_axis_km, body="Earth"):
    """
    Vis-viva equation: v = sqrt( GM * (2/r - 1/a) )
    Gives orbital speed at ANY point on an elliptical orbit. Circular
    velocity is just the special case where r equals a.
    """
    gm, body_radius = _get_body_constants(body)
    validate_radius(radius_km, body_radius)
    return math.sqrt(gm * (2 / radius_km - 1 / semi_major_axis_km))


def apoapsis_periapsis_velocities(periapsis_alt_km, apoapsis_alt_km, body="Earth"):
    """
    Returns velocity at periapsis and at apoapsis for an elliptical orbit.

    >>> CONCEPT: Tuples as return values.
    A function can only return once, but that one "thing" can be a tuple --
    a fixed-size bundle of values. Here we return (v_periapsis, v_apoapsis)
    so the caller gets both numbers back together, in a known, fixed order.
    """
    _, body_radius = _get_body_constants(body)
    r_p = periapsis_alt_km + body_radius
    r_a = apoapsis_alt_km + body_radius
    a = semi_major_axis(periapsis_alt_km, apoapsis_alt_km, body)

    v_periapsis = vis_viva_velocity(r_p, a, body)
    v_apoapsis = vis_viva_velocity(r_a, a, body)
    return (v_periapsis, v_apoapsis)  # fastest point, slowest point


def elliptical_orbit_period(periapsis_alt_km, apoapsis_alt_km, body="Earth"):
    """
    Kepler's third law generalizes cleanly: T = 2*pi*sqrt(a^3 / GM)
    where a is the semi-major axis. This also works for circular orbits
    (where periapsis == apoapsis, so a == r).
    """
    gm, _ = _get_body_constants(body)
    a = semi_major_axis(periapsis_alt_km, apoapsis_alt_km, body)
    return 2 * math.pi * math.sqrt(a ** 3 / gm)


# ---------------------------------------------------------------------------
# DISPLAY HELPER
# ---------------------------------------------------------------------------
def seconds_to_hms(seconds):
    """Convert a duration in seconds into a readable 'Hh Mm Ss' string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


# ---------------------------------------------------------------------------
# INTERACTIVE MENU
# ---------------------------------------------------------------------------
def run_circular_orbit_report(body):
    try:
        altitude = float(input(f"  Altitude above {body}'s surface (km): "))
        v = circular_orbital_velocity(altitude, body)
        t = orbital_period(altitude, body)
        print(f"\n  Circular orbital velocity: {v:.3f} km/s")
        print(f"  Orbital period:            {seconds_to_hms(t)}\n")
    except ValueError as e:
        # >>> CONCEPT: try/except
        # float(input(...)) crashes the whole program if the user types
        # something that isn't a number (e.g. "abc"). Wrapping it in
        # try/except catches that -- and our own OrbitInputError, which
        # IS a ValueError -- and prints a clean message instead of a
        # scary traceback.
        print(f"\n  Input error: {e}\n")


def run_elliptical_orbit_report(body):
    try:
        peri = float(input(f"  Periapsis altitude above {body}'s surface (km): "))
        apo = float(input(f"  Apoapsis altitude above {body}'s surface (km): "))

        v_p, v_a = apoapsis_periapsis_velocities(peri, apo, body)  # unpack tuple
        e = eccentricity(peri, apo, body)
        a = semi_major_axis(peri, apo, body)
        t = elliptical_orbit_period(peri, apo, body)

        print(f"\n  Semi-major axis:        {a:.1f} km")
        print(f"  Eccentricity:           {e:.4f}")
        print(f"  Velocity at periapsis:  {v_p:.3f} km/s  (fastest point)")
        print(f"  Velocity at apoapsis:   {v_a:.3f} km/s  (slowest point)")
        print(f"  Orbital period:         {seconds_to_hms(t)}\n")
    except ValueError as e:
        print(f"\n  Input error: {e}\n")


def main():
    print("=" * 60)
    print("  PROJECT 2 -- ORBITAL PARAMETER CALCULATOR")
    print("=" * 60)

    while True:
        print("\nAvailable bodies:", ", ".join(CELESTIAL_BODIES.keys()))
        body = input("Choose a body (or 'q' to quit): ").strip()
        if body.lower() == "q":
            break

        print("\n  1) Circular orbit (single altitude)")
        print("  2) Elliptical orbit (periapsis + apoapsis)")
        choice = input("  Choose orbit type: ").strip()

        if choice == "1":
            run_circular_orbit_report(body)
        elif choice == "2":
            run_elliptical_orbit_report(body)
        else:
            print("  Not a valid choice -- enter 1 or 2.\n")


# ---------------------------------------------------------------------------
# SELF-TEST -- VALIDATED AGAINST REAL ORBITS
# Run this file directly to see these checks before using the interactive
# menu. Same approach as P1: calculate, then compare to known real numbers.
# ---------------------------------------------------------------------------
def run_self_tests():
    print("Running validation checks against known real orbits...\n")

    # --- ISS: near-circular LEO, ~408 km altitude ---
    iss_alt = 408
    v_iss = circular_orbital_velocity(iss_alt, "Earth")
    t_iss = orbital_period(iss_alt, "Earth")
    print(f"ISS (~{iss_alt} km circular):")
    print(f"  Calculated velocity: {v_iss:.3f} km/s   (real-world: ~7.66 km/s)")
    print(f"  Calculated period:   {seconds_to_hms(t_iss)}   (real-world: ~92.7 min)\n")

    # --- GEO: geostationary circular orbit, ~35,786 km altitude ---
    geo_alt = 35786
    v_geo = circular_orbital_velocity(geo_alt, "Earth")
    t_geo = orbital_period(geo_alt, "Earth")
    print(f"GEO (~{geo_alt} km circular):")
    print(f"  Calculated velocity: {v_geo:.3f} km/s   (real-world: ~3.07 km/s)")
    print(f"  Calculated period:   {seconds_to_hms(t_geo)}   (real-world: ~23h 56m, one sidereal day)\n")

    # --- GTO: Geostationary Transfer Orbit, ~200 km perigee / ~35,786 km apogee ---
    gto_peri, gto_apo = 200, 35786
    v_p, v_a = apoapsis_periapsis_velocities(gto_peri, gto_apo, "Earth")
    e = eccentricity(gto_peri, gto_apo, "Earth")
    t_gto = elliptical_orbit_period(gto_peri, gto_apo, "Earth")
    print(f"GTO ({gto_peri} km x {gto_apo} km transfer orbit):")
    print(f"  Eccentricity:          {e:.3f}   (real-world: ~0.73)")
    print(f"  Velocity at perigee:   {v_p:.3f} km/s   (real-world: ~10.2 km/s)")
    print(f"  Velocity at apogee:    {v_a:.3f} km/s   (real-world: ~1.6 km/s)")
    print(f"  Transfer period:       {seconds_to_hms(t_gto)}   (real-world: ~10.5h)\n")

    print("=" * 60)


if __name__ == "__main__":
    run_self_tests()
    main()
