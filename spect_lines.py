import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

folder = 'data9'
title = 'fullspec00000'
data_file = f'{folder}/clean_data/{title}.csv'

data = pd.read_csv(data_file)

print(data)

volt_laser = data['volt_laser'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()
volt_ld = data['volt_ld'].to_numpy()

mask = (volt_piezo >= 1.25) & (volt_piezo <= 5.11)
piezo_restricted = volt_piezo[mask]
laser_restricted = volt_laser[mask]


def doppler_envelope(x, slope, intercept, scale, mean, sigma):
    return slope * x + intercept + scale * np.exp(-((x - mean)**2) / (2 * sigma**2))


param_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                [0, np.inf, 0, np.inf, np.inf])

popt, pcov = curve_fit(doppler_envelope, piezo_restricted,
                       laser_restricted, bounds=param_bounds)
print(popt)

vp = np.linspace(1.25, 5.11, 500)
fit = doppler_envelope(vp, *popt)

plt.figure()
plt.plot(piezo_restricted, laser_restricted, label='Laser intensity',
         color='blue', markersize=5, marker='.')
plt.plot(vp, fit, label='Fit result',
         color='red', markersize=5, marker='.')
plt.xlabel('Volt Piezo [V]')
plt.ylabel('Volt Laser [V]')
plt.title('Volt piezo vs Laser intensity')
plt.grid()
plt.legend()

just_peaks = laser_restricted - doppler_envelope(piezo_restricted, *popt)
peaks, _ = find_peaks(just_peaks, height=0.02, distance=100)

x_peaks = piezo_restricted[peaks]
y_peaks = just_peaks[peaks]

print(x_peaks)


plt.figure()
plt.plot(piezo_restricted, just_peaks, label='Peaks without doppler',
         color='blue', markersize=3, linewidth=2, marker='.')
plt.scatter(x_peaks, y_peaks, label='Peaks',
         color='red', s=20, marker='x')
plt.xlabel('Volt Piezo [V]')
plt.ylabel('Laser peaks [V]')
plt.title('Subtracting fit of doppler envelope from data')
plt.grid()
plt.legend()
plt.show()