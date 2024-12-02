import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''putting here all the possible functions'''


def transmission(x, slope, intercept, trans1, scale1, sigma1, trans2, scale2, sigma2, trans3, scale3, sigma3, trans4, scale4, sigma4):
    return (slope * x + intercept +
            trans1 * np.exp(-scale1 * np.exp(-((x - peaks['freq'][0])**2) / (2 * sigma1**2))) +
            trans2 * np.exp(-scale2 * np.exp(-((x - peaks['freq'][2])**2) / (2 * sigma2**2))) +
            trans3 * np.exp(-scale3 * np.exp(-((x - peaks['freq'][3])**2) / (2 * sigma3**2))) +
            trans4 * np.exp(-scale4 * np.exp(-((x - peaks['freq'][4])**2) / (2 * sigma4**2))))


def transmission_no_off(x, trans1, scale1, sigma1, trans2, scale2, sigma2, trans3, scale3, sigma3, trans4, scale4, sigma4):
    return (trans1 * np.exp(-scale1 * np.exp(-((x - peaks['freq'][0])**2) / (2 * sigma1**2))) +
            trans2 * np.exp(-scale2 * np.exp(-((x - peaks['freq'][2])**2) / (2 * sigma2**2))) +
            trans3 * np.exp(-scale3 * np.exp(-((x - peaks['freq'][3])**2) / (2 * sigma3**2))) +
            trans4 * np.exp(-scale4 * np.exp(-((x - peaks['freq'][4])**2) / (2 * sigma4**2))))


def transmission_temp(x, slope, intercept, temp, trans1, scale1, trans2, scale2, trans3, scale3, trans4, scale4):
    return (slope * x + intercept +
            trans1 * np.exp(-scale1 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[0] * (x - peaks['freq'][0])**2) / (peaks['freq'][0]**2 * 2 * kb * temp)) +
            trans2 * np.exp(-scale2 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[2] * (x - peaks['freq'][2])**2) / (peaks['freq'][2]**2 * 2 * kb * temp)) +
            trans3 * np.exp(-scale3 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[3] * (x - peaks['freq'][3])**2) / (peaks['freq'][3]**2 * 2 * kb * temp)) +
            trans4 * np.exp(-scale4 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[4] * (x - peaks['freq'][4])**2) / (peaks['freq'][4]**2 * 2 * kb * temp)))


'''reading data'''

data = pd.read_csv('data9/clean_data/Fixed_ld00000_frequencies_cropped.csv')

frequencies = data['frequencies'].to_numpy()
photodiode = data['photodiode'].to_numpy()

peaks = pd.read_csv('data9/clean_data/Fixed_ld00000_peaks_fit.csv')


'''first try: educated guesses, the best try for now'''

print('first try, educated guesses')

opt_slope = (photodiode[-1]-photodiode[0])/(frequencies[-1]-frequencies[0])
opt_intercept = photodiode[0] - opt_slope * frequencies[0]

# including the cross peak, will remove it later
kb = 1.38e-23  # J/K
T = 338  # K
rb_mass = 1.67e-27 * np.array([85, 85, 85, 87, 87])   # kg
c = 3e8
opt_sigma = np.sqrt(kb * T/rb_mass)*peaks['freq']/c

print(f'opt_slope: {opt_slope}')
print(f'opt_intercept: {opt_intercept}')
print(f'opt_sigma:\n{opt_sigma}')


p0 = [opt_slope, opt_intercept,  # linear part
      1.7, 0.7, opt_sigma[0],  # first transition
      1.7, 0.7, opt_sigma[2],  # second transition
      1.3, 0.2, opt_sigma[3],  # third transition
      1.3, 0.2, opt_sigma[4]]  # fourth transition

lower_bounds = [-np.inf, 0,  # linear part
                0, 0, 0,  # first transition
                0, 0, 0,  # second transition
                0, 0, 0,  # third transition
                0, 0, 0]  # fourth transition

upper_bounds = [0, np.inf,  # linear part
                4, 1, 1e9,  # first transition
                4, 1, 1e9,  # second transition
                4, 1, 1e9,  # third transition
                4, 1, 1e9]  # fourth transition

popt, pcov = curve_fit(transmission, xdata=frequencies,
                       ydata=photodiode, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)

print(f'first fit params:\n{popt}')

f = np.linspace(min(frequencies), max(frequencies), 500)
pd_fit = transmission(f, *popt)
line = popt[0] * f + popt[1]

plt.figure()
plt.scatter(frequencies, photodiode, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit, label='Fit result',
         color='red', linewidth=2)
plt.plot(f, line, label='offset line',
         color='green', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings, without peaks, guessing parameters')
plt.grid()
plt.legend()
plt.savefig('data9/figures/temperature/temp_fit_guess.pdf')


'''second try: educated guesses, without the linear offset'''

print('\nsecond try, no linear offset')

print(f'opt_sigma:\n{opt_sigma}')


p0_no_off = [1.7, 0.7, opt_sigma[0],  # first transition
             1.7, 0.7, opt_sigma[2],  # second transition
             1.3, 0.2, opt_sigma[3],  # third transition
             1.3, 0.2, opt_sigma[4]]  # fourth transition

lb_no_off = [0, 0, 0,  # first transition
             0, 0, 0,  # second transition
             0, 0, 0,  # third transition
             0, 0, 0]  # fourth transition

ub_no_off = [4, 1, 1e9,  # first transition
             4, 1, 1e9,  # second transition
             4, 1, 1e9,  # third transition
             4, 1, 1e9]  # fourth transition

popt_no_off, pcov_no_off = curve_fit(transmission_no_off, xdata=frequencies,
                                     ydata=photodiode, p0=p0_no_off, bounds=(lb_no_off, ub_no_off), maxfev=10000)

print(f'no offset fit params:\n{popt_no_off}')

pd_fit_no_off = transmission_no_off(f, *popt_no_off)

plt.figure()
plt.scatter(frequencies, photodiode, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit_no_off, label='Fit result',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings, without peaks, no linear offset')
plt.grid()
plt.legend()
plt.savefig('data9/figures/temperature/temp_fit_no_off.pdf')


'''third try: using parameter estimates from old fit'''

print('\nthird try, old fit parameters')

# copying previous result by hand
# it's not pretty, but i'm too exhausted to do it properly
df_dt = 140016965102.61118
conversion_intercept = 377110868880333.2

prev_slope = -3.30733868
prev_intercept = 1.41595671

# first "column" contains transN, second scaleN, third meanN, fourth sigmaN
previous_results_fit = [3.65285570e-01,  3.27978827e+00, -1.25644636e-02,  1.98361890e-03,
                        4.06747919e-01,  1.88628830e+00, -1.11875135e-02,  1.54480838e-03,
                        -1.43183832e-01, -5.15745291e-01,  2.71607023e-03,  1.72734503e-03,
                        -6.32830282e-01, -3.28447149e-01,  8.13065853e-03,  1.73946393e-03]

# convering old offset (computed with time on the x axis) into an offset for frequencies
new_slope = prev_slope / df_dt
new_int = -prev_slope * conversion_intercept / df_dt + prev_intercept

# sigma has to be scaled, other parameters don't
indices_sigma = [3, 7, 11, 15]
for i in indices_sigma:
    previous_results_fit[i] *= df_dt

# rmeoving means as we are already providing it in the function
previous_results_fit.pop(14)
previous_results_fit.pop(10)
previous_results_fit.pop(6)
previous_results_fit.pop(2)

print(f'new_slope: {new_slope}')
print(f'new_intercept: {new_int}')
print(f'other params guesses from old: {previous_results_fit}')

p0_old = [new_slope, new_int] + previous_results_fit

popt_old, pcov_old = curve_fit(transmission, xdata=frequencies,
                               ydata=photodiode, p0=p0_old, maxfev=10000)

print(f'fit params based on old fit:\n{popt_old}')

pd_fit_old = transmission(f, *popt_old)
line_old = popt_old[0] * f + popt_old[1]

plt.figure()
plt.scatter(frequencies, photodiode, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit_old, label='Fit result',
         color='red', linewidth=2)
plt.plot(f, line_old, label='offset line',
         color='green', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings, without peaks, based on old fit')
plt.grid()
plt.legend()
plt.savefig('data9/figures/temperature/temp_fit_old.pdf')


'''\nfourth try: imposing sigma to temperature relation'''

print('fourth try, direct fit of temperature parameter')

p0_temp = [opt_slope, opt_intercept, 338,  # linear part + temperature
           1.7, 0.7 * np.sqrt(T),  # first transition
           1.7, 0.7 * np.sqrt(T),  # second transition
           1.3, 0.2 * np.sqrt(T),  # third transition
           1.3, 0.2 * np.sqrt(T)]  # fourth transition

lower_bounds_temp = [-np.inf, 0, 273,  # linear part + temperature
                     0, 0,  # first transition
                     0, 0,  # second transition
                     0, 0,  # third transition
                     0, 0]  # fourth transition

upper_bounds_temp = [0, np.inf, np.inf,  # linear part + temperature
                     4, 1 * np.sqrt(T),  # first transition
                     4, 1 * np.sqrt(T),  # second transition
                     4, 1 * np.sqrt(T),  # third transition
                     4, 1 * np.sqrt(T)]  # fourth transition

popt_temp, pcov_temp = curve_fit(transmission_temp, xdata=frequencies,
                                 ydata=photodiode, p0=p0_temp, bounds=(lower_bounds_temp, upper_bounds_temp), maxfev=10000)

print('temperature fit params:')
print(popt_temp)

pd_fit_temp = transmission_temp(f, *popt_temp)
line_temp = popt_temp[0] * f + popt_temp[1]

plt.figure()
plt.scatter(frequencies, photodiode, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit_temp, label='Fit result',
         color='red', linewidth=2)
'''plt.plot(f, line_temp, label='offset line',
         color='green', linewidth=2)'''
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings, fitting temperature')
plt.grid()
plt.legend()
plt.savefig('data9/figures/temperature/temp_fit_temp.pdf')
plt.show()
