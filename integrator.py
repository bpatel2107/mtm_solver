#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import newton, minimize
import matplotlib.pyplot as plt
from scipy.special import j0
import scipy

e_charge = 1.6e-19
mu0 = 4 * np.pi * 1e-7
electron_mass = 9.1e-31
epsilon0 = 8.854e-12
deuterium_mass = 2 * 1.67e-27

phi_switch = 0

def lower_bound_xpar(xperp):
    return -np.inf

def upper_bound_xpar(xperp):
    return np.inf




def integrand_1(xpar,
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
              bessel,
              q):

    v_thermal = np.sqrt(2 * te / electron_mass)

    epsilon = r / R

    # x = v / vth
    x = np.sqrt(xpar ** 2 + xperp ** 2)

    # Maxwellian
    f0 = np.exp(-1 * x ** 2) * (j0(-ky*xperp*v_thermal * electron_mass / (e_charge*B))) ** 2

    omega_star = -1 * ky * te / (e_charge * B * R)
    omega_star_x = omega_star * (gne + gte * (x ** 2 - 1.5))


    omega_drift = 2 * ky * v_thermal / R ** 2 * (xpar ** 2 + 0.5 * xperp ** 2)
    omega_drift_x = omega_drift * 0.0
    omega_drift_x = 2 * ky * larmor_radius * sound_speed * r / R**2 * (1 - 1/q**2)

    
    nu_v = 1j * nu / epsilon / (x**3)  

    pre_factor_1 = 2 * ne * e_charge**2 * v_thermal**2 / (np.sqrt(np.pi) * te)

    integrand_1 = pre_factor_1 * xpar ** 2 * xperp * f0 * (omega - omega_star_x) / (
            omega - k_parallel * xpar - omega_drift_x + nu_v)

    # Return only real part for integration
    return integrand_1




def integrand_2(xpar,
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
              bessel,
              q):

    v_thermal = np.sqrt(2 * te / electron_mass)

    epsilon = r / R

    # x = v / vth
    x = np.sqrt(xpar ** 2 + xperp ** 2)

    # Maxwellian
    f0 = np.exp(-1 * x ** 2) * (j0(-ky*xperp*v_thermal * electron_mass / (e_charge*B))) ** 2

    omega_star = -1 * ky * te / (e_charge * B * R)
    omega_star_x = omega_star * (gne + gte * (x ** 2 - 1.5))

    omega_drift = 2 * ky * v_thermal / R ** 2 * (xpar ** 2 + 0.5 * xperp ** 2)
    omega_drift_x = omega_drift * 0.0
    omega_drift_x = 2 * ky * larmor_radius * sound_speed * r / R**2 * (1 - 1/q**2)


    nu_v = 1j * nu / epsilon / (x**3)  

    pre_factor_2 = 2 / np.sqrt(np.pi)

    integrand_2 = pre_factor_2 * xperp * f0 * (omega - omega_star_x) / (
            omega - k_parallel * xpar - omega_drift_x + nu_v)

    # Return only real part for integration
    return integrand_2




def integrand_3(xpar,
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
              bessel,
              q):

    v_thermal = np.sqrt(2 * te / electron_mass)

    epsilon = r / R

    # x = v / vth
    x = np.sqrt(xpar ** 2 + xperp ** 2)

    # Maxwellian
    f0 = np.exp(-1 * x ** 2) * (j0(-ky*xperp*v_thermal * electron_mass / (e_charge*B))) ** 2

    omega_star = -1 * ky * te / (e_charge * B * R)
    omega_star_x = omega_star * (gne + gte * (x ** 2 - 1.5))

    omega_drift = 2 * ky * v_thermal / R ** 2 * (xpar ** 2 + 0.5 * xperp ** 2)
    omega_drift_x = omega_drift * 0.0
    omega_drift_x = 2 * ky * larmor_radius * sound_speed * r / R**2 * (1 - 1/q**2)

    nu_v = 1j * nu / epsilon / (x**3)  

    pre_factor_3 = 2 * v_thermal / np.sqrt(np.pi)

    integrand_3 = pre_factor_3 * xpar * xperp * f0 * (omega - omega_star_x) / (
            omega - k_parallel * xpar - omega_drift_x + nu_v)

    # Return only real part for integration
    return integrand_3




def integrand_component_1(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q, func):

    return func(integrand_1(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q))

def integrand_component_2(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q, func):

    return func(integrand_2(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q))

def integrand_component_3(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q, func):

    return func(integrand_3(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q))


# In[7]:


def mtm_integral_1(*args):

    real_args = args + (np.real,)
    real_integral = dblquad(integrand_component_1, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=real_args)  
    imag_args = args + (np.imag,)
    imag_integral = dblquad(integrand_component_1, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=imag_args)
    
    complex_integral = real_integral[0] + 1j * imag_integral[0]
    complex_error = real_integral[1] + 1j * imag_integral[1]

    return complex_integral, complex_error

def mtm_integral_2(*args):

    real_args = args + (np.real,)
    real_integral = dblquad(integrand_component_2, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=real_args)  
    imag_args = args + (np.imag,)
    imag_integral = dblquad(integrand_component_2, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=imag_args)
    
    complex_integral = real_integral[0] + 1j * imag_integral[0]
    complex_error = real_integral[1] + 1j * imag_integral[1]

    return complex_integral, complex_error

def mtm_integral_3(*args):

    real_args = args + (np.real,)
    real_integral = dblquad(integrand_component_3, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=real_args)  
    imag_args = args + (np.imag,)
    imag_integral = dblquad(integrand_component_3, 0, np.inf, lower_bound_xpar, upper_bound_xpar, args=imag_args)
    
    complex_integral = real_integral[0] + 1j * imag_integral[0]
    complex_error = real_integral[1] + 1j * imag_integral[1]

    return complex_integral, complex_error


def minimise_func(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q, kperp):

    integral_1, error_1 =  mtm_integral_1(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q)
    integral_2, error_2 =  mtm_integral_2(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q)
    integral_3, error_3 =  mtm_integral_3(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q)

    if phi_switch != 0:
        # need to check signs
        minimse_value = ((ne * (e_charge ** 2) / te) * integral_3 ** 2 ) - (1 + (epsilon0 * te / (ne * e_charge ** 2)) * kperp ** 2 - integral_2) * (integral_1 + kperp ** 2 / mu0)    
    elif phi_switch == 0: 
        minimse_value = (integral_1 + kperp ** 2 / mu0)

    

    return minimse_value




if __name__=='__main__':

    
 # STEP Parameters
    r = 1.027
    R = 2.775
    q = 4.3
    shear = 0.74
    bunit = 7.52
    btor = 2.162

    gte = 4.9578
    gne = 0.7620

    ne = 15.1e19
    te = 12.173e3 * e_charge
    rho = 0.6627

    # NSTX Parameters
    r = 0.37
    R = 0.94
    q = 1.7
    shear = 1.7
    bunit = 0.666
    btor = 0.35

    gte = 4.1
    gne = 0.0

    ne = 6e19
    te = 0.45 * 1e3 * e_charge
    rho = 0.6

    minor_radius = r / rho


    # CGYRO units
    sound_speed = np.sqrt(te / deuterium_mass )
    ion_freq = e_charge * bunit / deuterium_mass
    larmor_radius = sound_speed / ion_freq
    rho_star = larmor_radius / minor_radius

    # Collisionality
    zeff = 1.0
    coolog = 24 - np.log(np.sqrt(ne* 1e-6) / (te / e_charge ) )

    nu = np.sqrt(2) * np.pi * ne * coolog * e_charge ** 4 / (
                 (4 * np.pi * epsilon0) ** 2 * electron_mass ** 0.5 *
                 te ** 1.5)
    
    nky = 10
    delta_n = 5
    kyrhos = np.empty(nky)
    omegas = np.empty(nky)
    gammas = np.empty(nky)
    nus = np.empty(nky)    


    integration_diff = np.empty(nky)

    for i in range(nky):

        n = (i + 1) * delta_n
        ky = n * q / r

        kx_over_ky = 1.0/0.2
        kperp = ky * np.sqrt(1**2 + kx_over_ky**2)
        omega_star = - (ky * te / (e_charge * bunit * R))

        k_parallel = 2 * shear / (R * q * kx_over_ky**2)

        # Initial guess for omega_guess
        omega_guess = omega_star * (gne + gte/2)
        #omega_guess = 0.0
        # Set up bessel function
        kyrhos[i] = ky * larmor_radius
        bessel = j0(kyrhos[i])

        print(f'kyrhos = {kyrhos[i]}')
        print(
            f"Initial guess: omega_guess = {np.real(omega_guess) / sound_speed * minor_radius:.2f}, gamma = {np.imag(omega_guess) / sound_speed * minor_radius:.2f}")

        # Run minimiser
        omega = newton(minimise_func, omega_guess, args=(ky, bunit, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q, kperp))
        print(f"{omega / sound_speed * minor_radius}\n")
    
        omegas[i] = np.real(omega / sound_speed * minor_radius)
        gammas[i] = np.imag(omega / sound_speed * minor_radius)
          
        
        
        
        # Test result
        minimiser = minimise_func(omega, ky, bunit, R, r, gte, gne, te, ne, k_parallel, nu, bessel, q, kperp)
        integration_diff[i] = abs(minimiser)

    # Save eigenvalues to file
    output = np.vstack((kyrhos, gammas,  omegas))
    np.savetxt('mtm-integrator_eigval.dat', np.transpose(output), header='MTM-integrator ky')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,7))

    
    #omegas = np.interp(kyrhos,ky)
    
    
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


