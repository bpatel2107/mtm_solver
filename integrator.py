import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import newton, minimize
import matplotlib.pyplot as plt


e_charge = 1.6e-19
mu0 = 4 * np.pi * 1e-7
electron_mass = 9.1e-31
epsilon0 = 8.854e-12
deuterium_mass = 2 * 1.67e-27

def lower_bound_xpar(xperp):
    return -np.inf

def upper_bound_xpar(xperp):
    return np.inf

def integrand(xpar,
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
    return integrand

def real_integrand(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu):

    inte = integrand(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu)

    return np.real(inte)


def imag_integrand(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu):

    inte = integrand(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu)

    return np.imag(inte)


def mtm_integral(*args):

    real_integral = dblquad(real_integrand, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=args)

    imag_integral = dblquad(imag_integrand, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=args)

    complex_integral = real_integral[0] + 1j * imag_integral[0]
    complex_error = real_integral[1] + 1j * imag_integral[1]

    return complex_integral, complex_error


def minimise_func(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, kperp):

    integral, error =  mtm_integral(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu)

    minimse_value = integral - kperp ** 2 / mu0

    return minimse_value


if __name__=='__main__':

    r = 1.027
    R = 2.775
    q = 4.3

    btor = 2.162
    bunit = 7.52
    gte = 4.9578
    gne = 0.7620

    ne = 15.2e19
    te = 12.173e3 * e_charge
    rho = 0.6627
    minor_radius = r / rho

    k_parallel = 0.0

    # Collisionality
    zeff = 1.0
    coolog = 24 - np.log(np.sqrt(ne * 1e-6) / te * e_charge )
    nu = 4 * (2 * np.pi)**0.5 * ne * coolog * e_charge**4 * zeff / (3 * (4 * np.pi * epsilon0)**2 * electron_mass**0.5 *
                                                                    te**1.5)
    # CGYRO units
    sound_speed = np.sqrt(te / deuterium_mass )
    ion_freq = e_charge * bunit / deuterium_mass
    larmor_radius = sound_speed / ion_freq
    rho_star = larmor_radius / minor_radius

    eta = gte / gne

    nky = 15
    delta_n = 10
    kyrhos = np.empty(nky)
    omegas = np.empty(nky)
    gammas = np.empty(nky)

    integration_diff = np.empty(nky)

    for n in range(nky):

        ky = delta_n * (n + 1) * q / r
        kperp = ky
        omega_star = (ky * te * gne / (e_charge * btor * R))

        # Initial guess for omega_guess
        omega_guess = omega_star * (1 + eta)

        kyrhos[n] = ky * larmor_radius
        print(f'kyrhos = {kyrhos[n]}')
        print(
            f"Initial guess: omega_guess = {np.real(omega_guess) / sound_speed * minor_radius:.2f}, gamma = {np.imag(omega_guess) / sound_speed * minor_radius:.2f}")

        # Run minimiser
        omega = newton(minimise_func, omega_guess, args=(ky, btor, R, r, gte, gne, te, ne, k_parallel, nu, kperp))
        print(f"{omega / sound_speed * minor_radius}\n")

        omegas[n] = np.real(omega / sound_speed * minor_radius)
        gammas[n] = np.imag(omega / sound_speed * minor_radius)

        # Test result
        minimiser = minimise_func(omega, ky, btor, R, r, gte, gne, te, ne, k_parallel, nu, kperp)
        integration_diff[n] = abs(minimiser)

    # Save eigenvalues to file
    output = np.vstack((kyrhos, gammas,  omegas))
    np.savetxt('mtm-integrator_eigval.dat', np.transpose(output), header='MTM-integrator ky')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,7))

    ax1.plot(kyrhos, gammas)
    ax1.set_ylabel(r'$c_s/a$')
    ax1.set_xlabel(r'$k_y\rho_s$')
    ax1.set_title(r'$\gamma$')

    ax2.plot(kyrhos, omegas)
    ax2.set_ylabel(r'$c_s/a$')
    ax2.set_xlabel(r'$k_y\rho_s$')
    ax2.set_title(r'$\omega$')

    plt.tight_layout()
    plt.show()

    plt.plot(kyrhos, integration_diff)
    plt.xlabel(r'$k_y\rho_s$')
    plt.title('Difference in integration')
    plt.yscale('log')
    plt.show()

