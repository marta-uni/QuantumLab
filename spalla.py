import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def transmission_no_off(x, trans, scale, sigma):
    return (trans * (np.exp(-scale * np.exp(-((x - peaks['freq'][4])**2) / (2 * sigma**2))))-1)

def transmission(x, slope, intercept, trans, scale, sigma):
    return ((slope * x + intercept)*(trans * np.exp(-scale * np.exp(-((x - peaks['freq'][4])**2) / (2 * sigma**2)))))



data = pd.read_csv('data9/clean_data/Fixed_ld00000_frequencies_cropped_2.csv')

frequencies = data['frequencies'].to_numpy()
photodiode = data['offset'].to_numpy()

peaks = pd.read_csv('data9/clean_data/Fixed_ld00000_peaks_fit.csv')

# including the cross peak, will remove it later
kb = 1.38e-23  # J/K
T = 338  # K
rb_mass = 1.67e-27 * np.array([85, 85, 85, 87, 87])   # kg
c = 3e8
opt_sigma = np.sqrt(kb * T/rb_mass)*peaks['freq']/c

print(f'opt_sigma:\n{opt_sigma}')

'''cutting at peak'''

print('\ncutting at peak')

p0 = [1.3, 0.2, opt_sigma[4]]

lower_bounds = [0, 0, 0]

upper_bounds = [4, 1, 1e9]

mask = (frequencies >= peaks['freq'][4]) & (frequencies <= 3e9 + 3.7711e14)
restricted_freq_peak = frequencies[mask]
restricted_pd_peak = photodiode[mask]

popt, pcov = curve_fit(transmission_no_off, xdata=restricted_freq_peak,
                       ydata=restricted_pd_peak, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)

print(f'fit params:\n{popt}')

f = np.linspace(min(restricted_freq_peak), max(restricted_freq_peak), 500)
pd_fit = transmission_no_off(f, *popt)

plt.figure()
plt.scatter(restricted_freq_peak, restricted_pd_peak, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit, label='Fit result',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings fit, cutting at peak')
plt.grid()
plt.legend()
plt.tight_layout()

temp_estimate = rb_mass[-1]/kb*(popt[2]*c/peaks['freq'][4])**2

print(f'temperature = {temp_estimate}')

'''cutting before peak'''

print('\ncutting before peak')

mask = (frequencies >= 11.7e9 + 3.771e14) & (frequencies <= 3e9 + 3.7711e14)
restricted_freq_1 = frequencies[mask]
restricted_pd_1 = photodiode[mask]

popt, pcov = curve_fit(transmission_no_off, xdata=restricted_freq_1,
                       ydata=restricted_pd_1, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)

print(f'fit params:\n{popt}')

f = np.linspace(min(restricted_freq_1), max(restricted_freq_1), 500)
pd_fit = transmission_no_off(f, *popt)

temp_estimate = rb_mass[-1]/kb*(popt[2]*c/peaks['freq'][4])**2

print(f'temperature = {temp_estimate}')

plt.figure()
plt.scatter(restricted_freq_1, restricted_pd_1, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit, label='Fit result',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings fit, cutting before peak')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

'''

print('\nincluding slope, cutting at peak')

opt_slope = (photodiode[-1]-photodiode[0])/(frequencies[-1]-frequencies[0])
opt_intercept = photodiode[0] - opt_slope * frequencies[0]

print(f'opt_slope: {opt_slope}')
print(f'opt_intercept: {opt_intercept}')

p0_slope = [opt_slope, opt_intercept] + p0
lb_slope = [-np.inf, 0] + lower_bounds
ub_slope = [0, np.inf] + upper_bounds

popt_slope, pcov_slope = curve_fit(transmission, xdata=restricted_freq_peak,
                       ydata=restricted_pd_peak, p0=p0_slope, bounds=(lb_slope, ub_slope), maxfev=10000)

print(f'fit params:\n{popt_slope}')

f = np.linspace(min(restricted_freq_peak), max(restricted_freq_peak), 500)
pd_fit_slope = transmission(f, *popt_slope)

plt.figure()
plt.scatter(restricted_freq_peak, restricted_pd_peak, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit_slope, label='Fit result',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings fit, including slope')
plt.grid()
plt.legend()
plt.tight_layout()


print('\nincluding slope, cutting before peak')

popt_slope_1, pcov_slope_1 = curve_fit(transmission, xdata=restricted_freq_1,
                       ydata=restricted_pd_1, p0=p0_slope, bounds=(lb_slope, ub_slope), maxfev=10000)

print(f'fit params:\n{popt_slope_1}')

f = np.linspace(min(restricted_freq_1), max(restricted_freq_1), 500)
pd_fit_slope = transmission(f, *popt_slope_1)

plt.figure()
plt.scatter(restricted_freq_1, restricted_pd_1, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit_slope, label='Fit result',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings fit, including slope')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()'''
