import numpy as np

def deltaM_model_ext(t, att, cbf, m0a, tau,
                 abv=0.0, att_a=0.7,
                 T1a=1.65, lambd=0.9, alpha=0.85 * 0.8):
    """
    This function extends the voxelwise model from the Chapell paper: equation [1] (tissue compartment) + [2] (arterial compartment)
    Parameter:
    - att: Arrival time tissue (s)
    - cbf: Cerebral blood flow (ml/100g/min)
    - m0a: M0a scaling (arbitrary units)
    - tau: Bolus duration (s)
    - abv: arterial blood volume
    - att_a: Arterial arrival time (s)
    """

    deltaM = np.zeros_like(t)

    # --- Equation [1] ---
    valid = t > att
    exp1 = np.exp(-(t[valid] - att) / T1a)
    exp2 = np.exp(-(t[valid] - att - tau) / T1a)

    deltaM[valid] += (
        (2 * alpha * cbf / 6000)
        * m0a
        * np.exp(-att / T1a)
        * (T1a / lambd)
        * (exp1 - exp2)
        / m0a
    )

    # --- Equation [2] ---
    valid_art = (t >= att_a) & (t <= att_a + tau)
    deltaM[valid_art] += (
        (2 * alpha * abv / lambd)
        * m0a
        * np.exp(-att_a / T1a)
    )

    return deltaM
