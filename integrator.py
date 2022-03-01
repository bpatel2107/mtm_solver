#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import newton, minimize
import matplotlib.pyplot as plt
from scipy.special import j0
import scipy

e_charge = 1.602176487e-19
mu0 = 4 * np.pi * 1e-7
electron_mass = 9.10938215e-31
speed_light = 299792458.0
epsilon0 = 1.0 / (speed_light**2 * mu0)
deuterium_mass = 2 * 1.672621638e-27

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
              nu,
              q,
              kperp):

    v_thermal = np.sqrt(2 * te / electron_mass)

    epsilon = r / R

    # x = v / vth
    x = np.sqrt(xpar ** 2 + xperp ** 2)

    # Maxwellian
    f0 = np.exp(-1 * x ** 2) * (j0(-kperp*xperp*v_thermal * electron_mass / (e_charge*B))) ** 2

    omega_star = -1 * ky * te / (e_charge * B * R)

    omega_star_x = omega_star * (gne + gte * (x ** 2 - 1.5))
    omega_drift_x = 2 * ky * larmor_radius * sound_speed * r / R**2 * (1 - 1/q**2) 
    
    nu_v = 1j * nu / epsilon / (x**3)
    
    v_alfven = B / np.sqrt(ne * mu0 * deuterium_mass)  * 0.0 

    pre_factor = 2 * ne * e_charge**2 * v_thermal**2 / (np.sqrt(np.pi) * te)

    integrand = pre_factor * (xpar ** 2 - xpar * k_parallel * v_alfven**2 / (omega * v_thermal)) * xperp * f0 * (omega - omega_star_x) / (
            omega - k_parallel * xpar  - omega_drift_x + nu_v)

    # Return only real part for integration
    return integrand



def integrand_component(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp, func):

    return func(integrand(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp))


def mtm_integral(*args):

    real_args = args + (np.real,)
    real_integral = dblquad(integrand_component, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=real_args)
    imag_args = args + (np.imag,)
    imag_integral = dblquad(integrand_component, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=imag_args)
    
    complex_integral = real_integral[0] + 1j * imag_integral[0]
    complex_error = real_integral[1] + 1j * imag_integral[1]

    return complex_integral, complex_error


def minimise_func(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp):

    integral, error =  mtm_integral(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp)

    minimse_value = (integral + kperp ** 2 / mu0)

    return minimse_value


if __name__=='__main__':

    """
    # STEP Parameters
    device = 'STEP'
    r = 1.027
    R = 2.775
    q = 4.3
    shear = 0.74
    bunit = 7.52
    btor = 2.162

    gte = 4.9578
    gne = 0.7620

    ne = 15.2e19
    te = 12.173e3 * e_charge
    rho = 0.6627
    """

    # NSTX Parameters
    device = 'NSTX'
    r = 0.369
    R = 0.942
    q = 1.70721
    shear = 1.70422
    bunit = 0.666
    btor = 0.35

    gte = 4.176
    gne = 0.0

    ne = 6.007e19
    te = 0.45 * 1e3 * e_charge
    rho = 0.6
    """
    # STEP Parameters
    device = 'SPR21'
    r = 0.982
    R = 3.017
    q = 2.9919
    shear = 0.78
    bunit = 7.064
    btor = 2.528

    gte = 4.024
    gne = 1.507

    ne = 1.889e20
    te = 8.5999e3 * e_charge
    rho = 0.6887
    """
    minor_radius = r / rho

    bfield = bunit
    
    # CGYRO units
    sound_speed = np.sqrt(te / deuterium_mass )
    ion_freq = e_charge * bfield / deuterium_mass
    larmor_radius = sound_speed / ion_freq
    rho_star = larmor_radius / minor_radius

    # Collisionality
    zeff = 1.0
    coolog = 24 - np.log(np.sqrt(ne* 1e-6) / (te / e_charge ) )

    nu = np.sqrt(2) * np.pi * ne * coolog * e_charge ** 4 / (
                 (4 * np.pi * epsilon0) ** 2 * electron_mass ** 0.5 *
                 te ** 1.5)

    #zlog = 37.8 - np.log(np.sqrt(ne) / (te * 1e-3 /e_charge))
    #zcf = (4 * np.sqrt(np.pi) / 3) * (e_charge / 
    #    (4 * np.pi * epsilon0)) ** 2 * 1e-3 * np.sqrt(e_charge / electron_mass * 1e-3)

    #nu = zcf * np.sqrt(2) * ne * zlog  * zeff / (te*1e-3/e_charge)**1.5
    
    nky = 4
    kyrhos = np.empty(nky)
    omegas = np.empty(nky)
    gammas = np.empty(nky)

    integration_diff = np.empty(nky)

    for i in range(nky):

        kyrhos[i] = (i+1) * 0.2
        ky = kyrhos[i]/ larmor_radius

        kx_over_ky = 1.0/0.2

        kperp = ky * np.sqrt(1**2 + kx_over_ky**2)
        #kperp = ky

        k_parallel = 2 * shear / (R * q * kx_over_ky**2)
        k_parallel = 0.0

        # Initial guess for omega_guess
        #omega_guess = -  kyrhos[i] * (gne + gte/2) * minor_radius / R * bfield / bfield * sound_speed / minor_radius

        omega_guess = -kyrhos[i] * (gne + gte / 2) * sound_speed / R
        omega_star = - (ky * te / (e_charge * bfield * R))
        omega_guess = omega_star * (gne + gte/2) * minor_radius / R * btor / bunit
        # Set up bessel function

        print(f'kyrhos = {kyrhos[i]}')
        print(
            f"Initial guess: omega_guess = {np.real(omega_guess) / sound_speed * minor_radius:.3f}, gamma = {np.imag(omega_guess) / sound_speed * minor_radius:.3f}")

        # Run minimiser
        omega = newton(minimise_func, omega_guess, args=(ky, bfield, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp))
        print(f"{omega / sound_speed * minor_radius}\n")
    
        omegas[i] = np.real(omega)
        gammas[i] = np.imag(omega)

        # Test result
        minimiser = minimise_func(omega, ky, bfield, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp)
        integration_diff[i] = abs(minimiser)

    # Save eigenvalues to file
    output = np.vstack((kyrhos, gammas,  omegas))
    np.savetxt(f'mtm-integrator-{device}_eigval.dat', np.transpose(output), header='MTM-integrator ky')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,7))

    
    #omegas = np.interp(kyrhos,ky)

    ax1.plot(kyrhos, gammas, label='In house code')

    ax1.set_ylabel(r'$c_s/a$')
    ax1.set_xlabel(r'$k_y\rho_s$')
    ax1.set_title(r'$\gamma$')
    ax1.legend()

    ax2.plot(kyrhos, omegas, label='In house code')
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


