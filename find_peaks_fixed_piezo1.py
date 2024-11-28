import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

folder = 'data9'
title = 'Fixed_piezo00001'
data_file = f'{folder}/clean_data/{title}.csv'

data = pd.read_csv(data_file)

volt_laser = data['volt_laser'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()
volt_ld = data['volt_ld'].to_numpy()

mask = (volt_ld >= -0.0625) & (volt_ld <= 0.089)
diode_restricted = volt_ld[mask]
laser_restricted = volt_laser[mask]


# just 2 gaussians as the third will not be recognized
def doppler_envelope(x, a, b, c, scale1, mean1, sigma1, scale2, mean2, sigma2):
    return (a * x**2 + b * x + c + scale1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))
            + scale2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2)))


param_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                [0, np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf])

popt, pcov = curve_fit(doppler_envelope, diode_restricted,
                       laser_restricted, bounds=param_bounds)
print(popt)

vl = np.linspace(min(volt_ld), max(volt_ld), 500)
fit = doppler_envelope(vl, *popt)

plt.figure()
plt.plot(volt_ld, volt_laser, label='Laser intensity',
         color='blue', markersize=5, marker='.')
plt.plot(vl, fit, label='Fit result',
         color='red', markersize=5, marker='')
plt.xlabel('Volt LD [V]')
plt.ylabel('Volt Laser [V]')
plt.title(f'Fitting {title}')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder}/figures/finding_peaks/{title}_fit.pdf')


just_peaks = volt_laser - doppler_envelope(volt_ld, *popt)
peaks_indices, _ = find_peaks(just_peaks, prominence=[
                              0.05, 0.3], height=0.045, distance=80)

peaks_indices = np.array(peaks_indices)

piezo_peaks = np.array(volt_piezo[peaks_indices])
laser_peaks = np.array(volt_laser[peaks_indices])
ld_peaks = np.array(volt_ld[peaks_indices])
y_peaks = np.array(just_peaks[peaks_indices])

print(ld_peaks)

plt.figure()
plt.plot(volt_ld, just_peaks, label='Peaks without doppler',
         color='blue', markersize=3, linewidth=2, marker='.')
plt.scatter(ld_peaks, y_peaks, label='Peaks',
            color='red', s=20, marker='x')
plt.xlabel('Volt ld [V]')
plt.ylabel('Laser peaks [V]')
plt.title(f'Rediduals {title}')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder}/figures/finding_peaks/{title}_residuals.pdf')
plt.show()

freq = [377108945610922.8, 377109126401922.8,
        377109307192922.8, 377111224766631.8]

# Saving data in clean_data folder
output_file = f'{folder}/clean_data/{title}_peaks.csv'
df = pd.DataFrame()
df['indices'] = peaks_indices
df['laser_peaks'] = laser_peaks
df['piezo_peaks'] = piezo_peaks
df['ld_peaks'] = ld_peaks
df['freq'] = freq

print(df)

df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
