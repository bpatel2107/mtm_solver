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

class calc_omega():

    def __init__(self,
                 ky_in   = 0.1,
                 rlt_in  = 3.909,
                 rln_in  = 0.290,
                 nu_in   = 0.03,
                 beta_in = 0.3,
                 q_in    = 3.0,
                 shat_in = 4.5):
        
        self.device = 'SPR21'
        self.r = 0.982
        self.R = 3.017
        self.q = q_in
        self.ky = ky_in
        
        self.shear = shat_in
        self.bunit = 10.064
        self.btor = 2.528
        self.beta = beta_in

        self.gte = rlt_in
        self.gne = rln_in

        self.ne = 1.889e20
        self.te = 8.5999e3 * e_charge
        self.rho = 0.6887
    
        self.minor_radius = self.r / self.rho

        self.bfield = self.bunit
    
        # CGYRO units
        # sound_speed = np.sqrt(te / deuterium_mass )
        #  ion_freq = e_charge * bfield / deuterium_mass
        # larmor_radius = sound_speed / ion_freq
        # rho_star = larmor_radius / minor_radius

        # GS2 units
        self.sound_speed = np.sqrt(self.te / deuterium_mass )
        self.ion_freq = e_charge * self.bfield / deuterium_mass
        self.larmor_radius = self.sound_speed / self.ion_freq
        self.rho_star = self.larmor_radius / self.minor_radius

        # Collisionality
        self.zeff = 1.0
        self.coolog = 24 - np.log(np.sqrt(self.ne* 1e-6) / (self.te / e_charge ) )

        #nu = np.sqrt(2) * np.pi * ne * coolog * e_charge ** 4 / (
        #nu       (4 * np.pi * epsilon0) ** 2 * electron_mass ** 0.5 *
        #         te ** 1.5)
        # explicit calculation is bypassed and scaled from input file
        self.nu = nu_in*self.sound_speed/self.minor_radius 

        #zlog = 37.8 - np.log(np.sqrt(ne) / (te * 1e-3 /e_charge))
        #zcf = (4 * np.sqrt(np.pi) / 3) * (e_charge / 
        #    (4 * np.pi * epsilon0)) ** 2 * 1e-3 * np.sqrt(e_charge / electron_mass * 1e-3)

        #nu = zcf * np.sqrt(2) * ne * zlog  * zeff / (te*1e-3/e_charge)**1.5
    
        self.nky = 1
        self.kyrhos = np.empty(self.nky)
        self.omegas = np.empty(self.nky)
        self.gammas = np.empty(self.nky)
        self.omega_guess = np.empty(self.nky)

        self.integration_diff = np.empty(self.nky)
        

    def calc(self):
     
        kyrhos = self.ky
        ky = kyrhos/ self.larmor_radius

        kx_over_ky = 1.0/0.2

        kperp = ky * np.sqrt(1**2 + kx_over_ky**2)
        #kperp = ky

        k_parallel = 2 * self.shear / (self.R * self.q * kx_over_ky**2)
        #k_parallel component currently commented out
        k_parallel = 0.0

        # Initial guess for omega_guess
        #omega_guess = -  kyrhos[i] * (gne + gte/2) * minor_radius / R * bfield / bfield * sound_speed / minor_radius

        omega_guess = -kyrhos * (self.gne + self.gte / 2) * self.sound_speed / self.minor_radius
        #omega_star = - (ky * te / (e_charge * bfield * R))
        #omega_guess = omega_star * (gne + gte/2) * minor_radius / R * btor / bunit
        # Set up bessel function

        print(f'kyrhos = {kyrhos}')
        print(
            f"Initial guess: omega_guess = {np.real(omega_guess) :.3f}, gamma = {np.imag(omega_guess) }")

        # Run minimiser
        omega = newton(self.minimise_func, omega_guess, args=(ky, self.bfield, self.R, self.r, self.gte, self.gne, self.te, self.ne, k_parallel, self.nu, self.q, kperp))

        return omega/ self.sound_speed * self.minor_radius

    def minimise_func(self, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp):

        integral, error =  self.mtm_integral(omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp)

        minimse_value = (integral + kperp ** 2 / mu0)

        return minimse_value

    def lower_bound_xpar(self,xperp):
        return -np.inf

    def upper_bound_xpar(self,xperp):
        return np.inf

    def integrand(self,
                  xpar,
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
        omega_drift_x = 2 * ky * self.larmor_radius * self.sound_speed * r / R**2 * (1 - 1/q**2) 
    
        nu_v = 1j * nu / epsilon / (x**3)
    
        #v_alfen explicitly
        #v_alfven = B / np.sqrt(ne * mu0 * deuterium_mass) 
        #v_alfven as scaled from the input beta
        v_alfven = self.sound_speed*np.sqrt(2)/np.sqrt(self.beta) 
       
        pre_factor = 2 * ne * e_charge**2 * v_thermal**2 / (np.sqrt(np.pi) * te)

        integrand = pre_factor * (xpar ** 2 - xpar * k_parallel * v_alfven**2 / (omega * v_thermal)) * xperp * f0 * (omega - omega_star_x) / (
            omega - k_parallel * xpar  - omega_drift_x + nu_v)

        # Return only real part for integration
        return integrand



    def integrand_component(self,xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp, func):

        return func(self.integrand(xpar, xperp, omega, ky, B, R, r, gte, gne, te, ne, k_parallel, nu, q, kperp))


    def mtm_integral(self,*args):

        real_args = args + (np.real,)
        real_integral = dblquad(self.integrand_component, 0, np.inf, self.lower_bound_xpar, self.upper_bound_xpar, args=real_args)
        imag_args = args + (np.imag,)
        imag_integral = dblquad(self.integrand_component, 0, np.inf, self.lower_bound_xpar, self.upper_bound_xpar, args=imag_args)
    
        complex_integral = real_integral[0] + 1j * imag_integral[0]
        complex_error = real_integral[1] + 1j * imag_integral[1]

        return complex_integral, complex_error



if __name__=='__main__':

    myrun = calc_omega(ky_in=0.4)
    omega = myrun.calc()
    print(omega)

    


