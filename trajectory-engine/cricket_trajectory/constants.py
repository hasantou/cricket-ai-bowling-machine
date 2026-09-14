"""Physical constants and standard cricket-ball dimensions (SI units)."""

# Standard gravitational acceleration, m/s^2
G = 9.81

# Pitch length between the two popping creases, metres (22 yards)
PITCH_LENGTH_M = 20.12

# Men's cricket ball, ICC standard:
#   mass 155.9-163.0 g, circumference 224-229 mm
BALL_MASS_KG = 0.160          # mid-range mass
BALL_CIRCUMFERENCE_M = 0.2265  # mid-range circumference
BALL_RADIUS_M = BALL_CIRCUMFERENCE_M / (2 * 3.141592653589793)

# Air at sea level, ~20 degrees C
AIR_DENSITY_KG_M3 = 1.225
AIR_DYNAMIC_VISCOSITY_PA_S = 1.81e-5

# km/h <-> m/s
def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh / 3.6


def ms_to_kmh(v_ms: float) -> float:
    return v_ms * 3.6
