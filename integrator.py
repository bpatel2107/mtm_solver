import numpy as np
from scipy.integrate import dblquad

e_charge = 1.6e-19
mu0 = 4 * np.pi * 1e-7
electron_mass = 9.1e-31
epsilon0 = 8.854e-12
deuterium_mass = 2 * 1.67e-27

def lower_bound_xpar(xperp):
    return -np.inf

def upper_bound_xpar(xperp):
    return np.inf


def mtm_integrand(xpar,
                  xperp,
                  omega,
                  ky,
                  B,
                  R,
                  r,
                  gte,
                  gne,
                  te,
                  ne,
                  k_parallel,
                  nu):

    v_thermal = np.sqrt(2 * te / electron_mass)

    epsilon = r / R

    # x = v / vth
    x = np.sqrt(xpar ** 2 + xperp ** 2)

    eta = gte / gne

    # Maxwellian
    f0 = np.exp(-x ** 2)

    omega_star = ky * te * gne / (e_charge * B * R)
    omega_star_x = omega_star * (1 + eta * (x ** 2 - 1.5))

    omega_drift = 2 * ky * v_thermal / R ** 2 * (xpar ** 2 + 0.5 * xperp ** 2)
    omega_drift_x = omega_drift * 0.0

    nu_v = 1j * nu / epsilon / x ** 3

    pre_factor = - 2 * ne * e_charge**2 * v_thermal**2 / (np.sqrt(np.pi) * te)

    integrand = pre_factor * xpar ** 2 * xperp * f0 * (omega - omega_star_x) / (
            omega - k_parallel * xpar - omega_drift_x + nu_v)

    # Return only real part for integration
    return np.real(integrand)

# STEP parameters
n = 5

r = 1.027
R = 2.775
q = 4.3

btor = 2.162
gte = 4.9578
gne = 0.7620

ky = n * q / r

ne = 15.2e19
te = 12.173e3 * e_charge
rho = 0.66
minor_radius = r / rho

sound_speed = np.sqrt(te / deuterium_mass )
ion_freq = e_charge * btor / deuterium_mass
rho_star = sound_speed / ion_freq

omega_star = (ky * te * gne / (e_charge * btor * R)) * (1 + gte/gne)

# Initial guess for omega
omega = + omega_star + 1j * omega_star / 7.2824

k_parallel = 0.0

# Collisionality
zeff = 1.0
coolog = 24 - np.log(np.sqrt(ne * 1e-6) / te * e_charge )
nu = 4 * (2 * np.pi)**0.5 * ne * coolog * e_charge**4 * zeff / (3 * (4 * np.pi * epsilon0)**2 *electron_mass**0.5 *                                                                te**1.5)

# Arguments in function
args = [omega, ky, btor, R, r, gte, gne, te, ne, k_parallel, nu]

# Perform integral
result = dblquad(mtm_integrand, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=args)

integral = result[0]
error = result[1]

difference = integral - ky**2 / mu0
print(f"{integral:.2e}, {ky**2/mu0:.2e}, {difference:.2e}")
