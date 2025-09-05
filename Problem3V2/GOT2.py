# Extended multi-beam thin-film model with amplitude roll-off (thickness spread, roughness, resolution)
# This script will:
#  - Use df3 (10°) and df4 (15°) if they exist in the environment; otherwise generate synthetic demos.
#  - Plot measured (or synthetic) vs model curves for a user-specified thickness d.
#  - Include three realism knobs: sigma_d (thickness spread), sigma_r (roughness), and instrument FWHM.
#
# Knobs to tweak (top of file):
#   d_um, n2_const, n3_const
#   sigma_d_um, sigma_theta_deg, sigma_r_nm, instr_fwhm_cm1
#
# Notes:
#  - Unpolarized reflectance = 0.5*(Rs + Rp)
#  - Dispersion and absorption are omitted for speed; can be added later if needed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- User parameters ----------------------
d_um = 8.0             # target thickness in micrometers
n1_const = 1.0         # ambient (air)
n2_const = 3.46        # epi layer (Si or SiC approx) - tweak as needed
n3_const = 3.50        # substrate (slightly different to create contrast)

# Realism knobs (set to >0 to activate effects)
sigma_d_um = 0.3      # thickness std dev (e.g., 0.03 µm = 30 nm). Drives high-ν amplitude roll-off
sigma_theta_deg = 0.1  # beam divergence / angle spread (deg)
sigma_r_nm = 0.3       # interface roughness rms (nm) -> Debye–Waller-like amplitude reduction
instr_fwhm_cm1 = 4.0   # instrument spectral FWHM in cm^-1 (Gaussian). Use 0 to disable

# Sampling for Gaussian averaging (7~9 is good; keep small for speed)
n_d_samples = 7
n_th_samples = 3
# -------------------------------------------------------------

def snell_theta(n1, n2, theta1_rad):
    s = n1*np.sin(theta1_rad)/n2
    s = np.clip(s, -1.0, 1.0)
    return np.arcsin(s)

def fresnel_coeffs_s(n1, n2, th1, th2):
    num = n1*np.cos(th1) - n2*np.cos(th2)
    den = n1*np.cos(th1) + n2*np.cos(th2)
    return num/den

def fresnel_coeffs_p(n1, n2, th1, th2):
    num = n2*np.cos(th1) - n1*np.cos(th2)
    den = n2*np.cos(th1) + n1*np.cos(th2)
    return num/den

def gauss_samples(mu, sigma, n):
    if sigma <= 0 or n == 1:
        return np.array([mu]), np.array([1.0])
    z = np.linspace(-2.0, 2.0, n)
    w = np.exp(-0.5*z**2)
    w /= w.sum()
    return mu + sigma*z, w

def airy_reflectance_unpolarized_avg(n1, n2, n3, theta1_rad, nu_cm1, d_um,
                                     sigma_d_um=0.0, n_d_samples=7,
                                     sigma_theta_deg=0.0, n_th_samples=1,
                                     sigma_r_nm=0.0):
    """
    Multi-beam Airy model with averaging over thickness and angle,
    plus Debye–Waller-like roughness attenuation of coherent amplitudes.
    """
    d_cm0 = d_um * 1e-4
    d_list, wd = gauss_samples(d_cm0, sigma_d_um*1e-4, n_d_samples)
    th0 = theta1_rad
    th_list, wth = gauss_samples(th0, np.deg2rad(sigma_theta_deg), n_th_samples)

    def rough_factor(theta_eff, sigma_r_nm):
        if sigma_r_nm <= 0:
            return 1.0
        sigma_r_cm = sigma_r_nm * 1e-7
        # amplitude attenuation factor (simple): exp[-(4πσ cosθ ν)^2]
        return np.exp(-(4*np.pi*sigma_r_cm*np.cos(theta_eff)*nu_cm1)**2)

    R_acc = np.zeros_like(nu_cm1, dtype=float)
    for di, wi in zip(d_list, wd):
        for th1_i, wj in zip(th_list, wth):
            th2 = snell_theta(n1, n2, th1_i)
            th3 = snell_theta(n2, n3, th2)

            r12_s = fresnel_coeffs_s(n1, n2, th1_i, th2)
            r12_p = fresnel_coeffs_p(n1, n2, th1_i, th2)
            r23_s = fresnel_coeffs_s(n2, n3, th2, th3)
            r23_p = fresnel_coeffs_p(n2, n3, th2, th3)

            # Apply roughness factor to interface amplitudes
            f12 = rough_factor(th2, sigma_r_nm)
            f23 = rough_factor(th2, sigma_r_nm)
            r12_s *= f12; r12_p *= f12
            r23_s *= f23; r23_p *= f23

            beta = 2*np.pi * nu_cm1 * n2 * di * np.cos(th2)
            e2ib = np.exp(2j*beta)
            r_s = (r12_s + r23_s*e2ib) / (1 + r12_s*r23_s*e2ib)
            r_p = (r12_p + r23_p*e2ib) / (1 + r12_p*r23_p*e2ib)
            R_acc += (wi*wj) * 0.5*(np.abs(r_s)**2 + np.abs(r_p)**2)

    return R_acc

def convolve_resolution(nu, R, fwhm_cm1=0.0):
    if fwhm_cm1 <= 0:
        return R
    sigma = fwhm_cm1/np.sqrt(8*np.log(2))
    dnu = np.median(np.diff(nu))
    if dnu <= 0 or np.isnan(dnu):
        return R
    half = int(np.ceil(4*sigma/dnu))
    k = np.arange(-half, half+1)*dnu
    g = np.exp(-0.5*(k/sigma)**2)
    g /= g.sum()
    return np.convolve(R, g, mode="same")

def build_model_from_df(df, theta_deg):
    nu = df.iloc[:,0].to_numpy(dtype=float)
    R_meas = df.iloc[:,1].to_numpy(dtype=float) / 100.0
    theta1 = np.deg2rad(theta_deg)
    R_model = airy_reflectance_unpolarized_avg(
        n1_const, n2_const, n3_const, theta1, nu, d_um,
        sigma_d_um=sigma_d_um, n_d_samples=n_d_samples,
        sigma_theta_deg=sigma_theta_deg, n_th_samples=n_th_samples,
        sigma_r_nm=sigma_r_nm
    )
    R_model = convolve_resolution(nu, R_model, instr_fwhm_cm1)
    return nu, R_meas, R_model

def plot_compare(df, theta_deg, title_suffix):
    nu, R_meas, R_model = build_model_from_df(df, theta_deg)
    plt.figure()
    plt.plot(nu, R_meas, label="Measured")
    plt.plot(nu, R_model, label=(
        f"Model (d={d_um:.3f} µm, σd={sigma_d_um*1e3:.0f} nm, "
        f"σr={sigma_r_nm:.1f} nm, res={instr_fwhm_cm1:.1f} cm⁻¹)"
    ))
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Reflectance")
    plt.title(f"Si/SiC Epi on Substrate – {theta_deg}° ({title_suffix})")
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
