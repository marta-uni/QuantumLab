import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

folder = 'data9'
title = 'fullspec00000'
data_file = f'{folder}/clean_data/{title}.csv'

data = pd.read_csv(data_file)

timestamp = data['timestamp'].to_numpy()
volt_laser = data['volt_laser'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()
volt_ld = data['volt_ld'].to_numpy()

mask = (volt_piezo >= -4.5)
piezo_restricted = volt_piezo[mask]
laser_restricted = volt_laser[mask]


def doppler_envelope(x, a, b, c, scale1, mean1, sigma1, scale2, mean2, sigma2, scale3, mean3, sigma3, scale4, mean4, sigma4, scale5, mean5, sigma5):
    return (a * x**2 + b * x + c + scale1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))
            + scale2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2))
            + scale3 * np.exp(-((x - mean3)**2) / (2 * sigma3**2))
            + scale4 * np.exp(-((x - mean4)**2) / (2 * sigma4**2))
            + scale5 * np.exp(-((x - mean5)**2) / (2 * sigma5**2)))


param_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -4.33, 0, -np.inf, -1.81, 0, -np.inf, 2.98, 0, -np.inf, 6.24, 0, -np.inf, 7.33, 0],
                [0, np.inf, np.inf, 0, -3.37, 1.4, 0, -0.49, 2.6, 0, 4.44, 1.8, 0, 7, 1, 0, 8.08, 1])

popt, pcov = curve_fit(doppler_envelope, piezo_restricted,
                       laser_restricted, bounds=param_bounds)
print(popt)

vp = np.linspace(-4.5, max(volt_piezo), 500)
fit = doppler_envelope(vp, *popt)

plt.figure()
plt.plot(volt_piezo, volt_laser, label='Laser intensity',
         color='blue', markersize=5, marker='.')
plt.plot(vp, fit, label='Fit result',
         color='red', markersize=5, marker='')
plt.xlabel('Volt Piezo [V]')
plt.ylabel('Volt Laser [V]')
plt.title(f'Fitting {title}')
plt.grid()
plt.legend()
# plt.savefig(f'{folder}/figures/finding_peaks/{title}_fit.pdf')


just_peaks = volt_laser - doppler_envelope(volt_piezo, *popt)
peaks_indices, _ = find_peaks(just_peaks, prominence=[0.015, 0.3], distance=50)

peaks_indices = np.array(peaks_indices)
# Correct peaks

indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 13, 15]
peaks_indices = peaks_indices[indices]

timestamp = np.array(timestamp[peaks_indices])
piezo_peaks = np.array(volt_piezo[peaks_indices])
laser_peaks = np.array(volt_laser[peaks_indices])
ld_peaks = np.array(volt_ld[peaks_indices])
y_peaks = np.array(just_peaks[peaks_indices])

print(piezo_peaks)

plt.figure()
plt.plot(volt_piezo, just_peaks, label='Peaks without doppler',
         color='blue', markersize=3, linewidth=2, marker='.')
plt.scatter(piezo_peaks, y_peaks, label='Peaks',
            color='red', s=20, marker='x')
plt.xlabel('Volt Piezo [V]')
plt.ylabel('Laser peaks [V]')
plt.title(f'Rediduals {title}')
plt.grid()
plt.legend()
plt.tight_layout()
# plt.savefig(f'{folder}/figures/finding_peaks/{title}_residuals.pdf')
plt.show()

freq = [377104390084020.94, 377104798412020.94, 377105206740020.94, 377105909878483.7, 377106090669483.7, 377106271460483.7,
        377108945610922.8, 377109126401922.8, 377109307192922.8, 377111224766631.8, 377112041422631.8]

# Saving data in clean_data folder
output_file = f'{folder}/clean_data/{title}_peaks.csv'
df = pd.DataFrame()
df['indices'] = peaks_indices
df['timestamp'] = timestamp
df['pd_peaks'] = laser_peaks
df['piezo_peaks'] = piezo_peaks
df['ld_peaks'] = ld_peaks
df['freq'] = freq

print(df)

df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
