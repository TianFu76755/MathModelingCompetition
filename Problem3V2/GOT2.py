# Quick validation demo for multi-beam thin-film model vs. measured data (df3/df4).
# Assumptions:
# - df3: angle = 10 degrees; columns: [wavenumber_cm1, reflectance_percent]
# - df4: angle = 15 degrees; columns: [wavenumber_cm1, reflectance_percent]
# - Unpolarized light -> average of s & p.
# - Simple (but editable) refractive indices for Silicon epi layer (n2) and substrate (n3).
# - No absorption (k≈0) and no instrument convolution for speed.
#
# You can tweak: d_um, n2_const, n3_const.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- User-tweakable parameters ----
d_um = 20.0            # thickness in micrometers (change this to test), default 5 µm
n1_const = 1.0        # air
n2_const = 3.46       # epi Si (approx., tweakable)
n3_const = 3.50       # substrate Si (slightly different to create contrast, tweakable)

# -----------------------------------

def snell_theta(n1, n2, theta1_rad):
    # Handle possible numerical drift
    s = n1*np.sin(theta1_rad)/n2
    s = np.clip(s, -1.0, 1.0)
    return np.arcsin(s)

def fresnel_coeffs_s(n1, n2, th1, th2):
    # s-polarized amplitude reflection coefficient
    num = n1*np.cos(th1) - n2*np.cos(th2)
    den = n1*np.cos(th1) + n2*np.cos(th2)
    return num/den

def fresnel_coeffs_p(n1, n2, th1, th2):
    # p-polarized amplitude reflection coefficient
    num = n2*np.cos(th1) - n1*np.cos(th2)
    den = n2*np.cos(th1) + n1*np.cos(th2)
    return num/den

def airy_reflectance_unpolarized(n1, n2, n3, theta1_rad, nu_cm1, d_um):
    """
    Single-film (medium 2) on substrate (medium 3) in ambient (medium 1).
    nu_cm1: numpy array of wavenumbers in cm^-1
    d_um: film thickness in micrometers
    """
    # Convert thickness to cm (since nu is in cm^-1)
    d_cm = d_um * 1e-4

    # Angles in each medium
    th2 = snell_theta(n1, n2, theta1_rad)
    th3 = snell_theta(n2, n3, th2)

    # Fresnel amplitude coefficients for both interfaces
    r12_s = fresnel_coeffs_s(n1, n2, theta1_rad, th2)
    r12_p = fresnel_coeffs_p(n1, n2, theta1_rad, th2)
    r23_s = fresnel_coeffs_s(n2, n3, th2, th3)
    r23_p = fresnel_coeffs_p(n2, n3, th2, th3)

    # Phase thickness beta(ν) = 2π * ν * n2 * d * cos(theta2)
    # Here we use constant n2 (no dispersion) for quick validation; can be extended easily.
    beta = 2*np.pi * nu_cm1 * n2 * d_cm * np.cos(th2)

    # Total amplitude reflection (Airy) for s and p
    exp_term = np.exp(2j*beta)
    r_s = (r12_s + r23_s*exp_term) / (1 + r12_s*r23_s*exp_term)
    r_p = (r12_p + r23_p*exp_term) / (1 + r12_p*r23_p*exp_term)

    # Unpolarized reflectance
    R = 0.5*(np.abs(r_s)**2 + np.abs(r_p)**2)
    return R

# Helper to build modeled curve aligned to a dataframe
def build_model_from_df(df, theta_deg):
    # Expect first col = wavenumber (cm^-1), second col = reflectance (%)
    nu = df.iloc[:,0].to_numpy(dtype=float)
    R_meas = df.iloc[:,1].to_numpy(dtype=float) / 100.0  # to fraction

    theta1 = np.deg2rad(theta_deg)
    R_model = airy_reflectance_unpolarized(
        n1_const, n2_const, n3_const, theta1, nu, d_um
    )
    return nu, R_meas, R_model

# ---- Generate plots ----
def plot_compare(df, theta_deg, title_suffix):
    nu, R_meas, R_model = build_model_from_df(df, theta_deg)

    plt.figure()
    plt.plot(nu, R_meas, label="Measured")
    plt.plot(nu, R_model, label=f"Model (d={d_um:.3f} µm)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Reflectance")
    plt.title(f"Si Epi on Si – {theta_deg}° ({title_suffix})")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    from Toolkit.ChineseSupport import show_chinese

    show_chinese()
    # 你工程里的读取方式
    from Data.DataManager import DM

    df1 = DM.get_data(1)
    df2 = DM.get_data(2)
    df3 = DM.get_data(3)
    df4 = DM.get_data(4)
    # Make two separate figures (required: one chart per figure)
    plot_compare(df3, 10, "df3")
    plot_compare(df4, 15, "df4")
