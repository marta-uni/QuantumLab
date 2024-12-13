import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler


def transmission_temp_no_off(x, ampl, scale1, scale2, temp):
    kb = 1.38e-23  # J/K
    rb_mass = 1.67e-27 * 87   # kg
    c = 3e8
    term1 = np.sqrt(rb_mass / (2 * kb * np.pi)) * c / (2 * np.pi)
    term2 = kb / (c**2 * rb_mass)
    return ampl * (np.exp(- term1 / np.sqrt(temp) * (scale1 / peaks['freq'][3] * np.exp(-(x - peaks['freq'][3])**2 / (2 * term2 * (peaks['freq'][3]**2) * temp)) +
                                                     scale2 / peaks['freq'][4] * np.exp(-(x - peaks['freq'][4])**2 / (2 * term2 * (peaks['freq'][4]**2) * temp)))) - 1)

def transmission_temp_no_off_scaled(x, ampl, scale1, scale2, temp):
    kb = 1.38e-23  # J/K
    rb_mass = 1.67e-27 * 87   # kg
    c = 3e8
    term1 = np.sqrt(rb_mass / (2 * kb * np.pi)) * c / (2 * np.pi)
    term2 = kb / (c**2 * rb_mass * scale_factor**2)
    return ampl * (np.exp(- term1 / np.sqrt(temp) * (scale1 / peaks['freq'][3] * np.exp(-(x - peaks['scaled_freq'][3])**2 / (2 * term2 * (peaks['freq'][3]**2) * temp)) +
                                                     scale2 / peaks['freq'][4] * np.exp(-(x - peaks['scaled_freq'][4])**2 / (2 * term2 * (peaks['freq'][4]**2) * temp)))) - 1)

data = pd.read_csv('data9/clean_data/Fixed_ld00000_frequencies_cropped_2.csv')

peaks = pd.read_csv('data9/clean_data/Fixed_ld00000_peaks_fit.csv')

frequencies = data['frequencies'].to_numpy()
photodiode = data['offset'].to_numpy()

# lower_mask = 5e8 + + 3.7711e14
lower_mask = peaks['freq'][4]
upper_mask = 3e9 + 3.7711e14

mask = (frequencies >= lower_mask) & (frequencies <= upper_mask)
restricted_freq = frequencies[mask]
restricted_pd = photodiode[mask]

lower_bounds = [0, 0, 0, 273]
upper_bounds = [np.inf, np.inf, np.inf, 500]

popt, pcov = curve_fit(transmission_temp_no_off, xdata=restricted_freq,
                       ydata=restricted_pd, bounds=(lower_bounds, upper_bounds), maxfev=10000)

print(f'amplitude:\t{popt[0]} +/- {np.sqrt(pcov[0,0])} V')
print(f'scale1:\t\t{popt[1]} +/- {np.sqrt(pcov[1,1])} m^3')
print(f'scale2:\t\t{popt[2]} +/- {np.sqrt(pcov[2,2])} m^3')
print(f'temperature:\t{popt[3]} +/- {np.sqrt(pcov[3,3])} K')

f = np.linspace(min(restricted_freq), max(restricted_freq), 500)
pd_fit = transmission_temp_no_off(f, *popt)

plt.figure()
plt.scatter(restricted_freq, restricted_pd, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit, label=f'Fit result, T$={popt[3]:.1f}\pm{np.sqrt(pcov[3,3]):.1f}$K',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.title('Photodiode readings fit, cutting at peak')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('data9/figures/temperature/temp_fit_shoulder.pdf')


print('\nTry scaling')

scaler = MinMaxScaler()

frequencies_reshaped = frequencies.reshape(-1, 1)
fs = scaler.fit_transform(frequencies_reshaped)
scaled_frequencies = fs.flatten()

scale_factor = scaler.data_max_ - scaler.data_min_
x_min = scaler.data_min_

peaks['scaled_freq'] = (peaks['freq'] - x_min) / scale_factor

mask = (scaled_frequencies >= ((lower_mask - x_min) / scale_factor)) & (
    scaled_frequencies <= ((upper_mask - x_min) / scale_factor))
restricted_freq_scaled = scaled_frequencies[mask]

lower_bounds = [0, 0, 0, 273]
upper_bounds = [np.inf, np.inf, np.inf, 500]

popt, pcov = curve_fit(transmission_temp_no_off_scaled, xdata=restricted_freq_scaled,
                       ydata=restricted_pd, bounds=(lower_bounds, upper_bounds), maxfev=10000)

print(f'amplitude:\t{popt[0]} +/- {np.sqrt(pcov[0,0])} V')
print(f'scale1:\t\t{popt[1]} +/- {np.sqrt(pcov[1,1])} m^3')
print(f'scale2:\t\t{popt[2]} +/- {np.sqrt(pcov[2,2])} m^3')
print(f'temperature:\t{popt[3]} +/- {np.sqrt(pcov[3,3])} K')

pd_fit = transmission_temp_no_off(f, *popt)

plt.figure()
plt.scatter(restricted_freq, restricted_pd, label='Data',
            color='blue', s=5, marker='.')
plt.plot(f, pd_fit, label='Fit result',
         color='red', linewidth=2)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Photodiode readings [V]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
