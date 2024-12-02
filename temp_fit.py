import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_csv('data9/clean_data/Fixed_ld00000_frequencies_cropped.csv')

frequencies = data['frequencies'].to_numpy()
photodiode = data['photodiode'].to_numpy()

peaks = pd.read_csv('data9/clean_data/Fixed_ld00000_peaks_fit.csv')

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
print(f'opt_sigma: {opt_sigma}')


def transmission(x, slope, intercept, trans1, scale1, sigma1, trans2, scale2, sigma2, trans3, scale3, sigma3, trans4, scale4, sigma4):
    return (slope * x + intercept +
            trans1 * np.exp(-scale1 * np.exp(-((x - peaks['freq'][0])**2) / (2 * sigma1**2))) +
            trans2 * np.exp(-scale2 * np.exp(-((x - peaks['freq'][2])**2) / (2 * sigma2**2))) +
            trans3 * np.exp(-scale3 * np.exp(-((x - peaks['freq'][3])**2) / (2 * sigma3**2))) +
            trans4 * np.exp(-scale4 * np.exp(-((x - peaks['freq'][4])**2) / (2 * sigma4**2))))


def transmission_temp(x, slope, intercept, temp, trans1, scale1, trans2, scale2, trans3, scale3, trans4, scale4):
    return (slope * x + intercept +
            trans1 * np.exp(-scale1 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[0] * (x - peaks['freq'][0])**2) / (peaks['freq'][0]**2 * 2 * kb * temp)) +
            trans2 * np.exp(-scale2 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[2] * (x - peaks['freq'][2])**2) / (peaks['freq'][2]**2 * 2 * kb * temp)) +
            trans3 * np.exp(-scale3 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[3] * (x - peaks['freq'][3])**2) / (peaks['freq'][3]**2 * 2 * kb * temp)) +
            trans4 * np.exp(-scale4 / np.sqrt(temp) * np.exp(-c**2 * rb_mass[4] * (x - peaks['freq'][4])**2) / (peaks['freq'][4]**2 * 2 * kb * temp)))


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

print('first fit params:')
print(popt)

f = np.linspace(min(frequencies), max(frequencies), 500)
pd_fit = transmission(f, *popt)
line = popt[0] * f + popt[1]

p0_temp = [opt_slope, opt_intercept, 338,  # linear part + temperature
           1.7, 0.7,  # first transition
           1.7, 0.7,  # second transition
           1.3, 0.2,  # third transition
           1.3, 0.2]  # fourth transition

lower_bounds_temp = [-np.inf, 0, 273,  # linear part + temperature
                     0, 0,  # first transition
                     0, 0,  # second transition
                     0, 0,  # third transition
                     0, 0]  # fourth transition

upper_bounds_temp = [0, np.inf, np.inf,  # linear part + temperature
                     4, 1,  # first transition
                     4, 1,  # second transition
                     4, 1,  # third transition
                     4, 1]  # fourth transition


popt_temp, pcov_temp = curve_fit(transmission_temp, xdata=frequencies,
                                 ydata=photodiode, p0=p0_temp, bounds=(lower_bounds_temp, upper_bounds_temp), maxfev=10000)

print('temperature fit params:')
print(popt_temp)

pd_temp_fit = transmission_temp(f, *popt_temp)
line_temp = popt_temp[0] * f + popt_temp[1]

plt.figure()
plt.scatter(frequencies, photodiode, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit, label='Fit result',
         color='red', linewidth=2)
plt.plot(f, line, label='offset line',
         color='green', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings, without peaks')
plt.grid()
plt.legend()
plt.savefig('data9/figures/temperature/temp_fit.png')
plt.show()
