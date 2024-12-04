import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
import numpy as np
from scipy.optimize import curve_fit
import fit_peaks as fp
from scipy.signal import find_peaks
import plotting_functions as pf
from sklearn.preprocessing import MinMaxScaler


def doppler_envelope_rough(x, slope, intercept, scale1, mean1, sigma1, scale2, mean2, sigma2, scale3, mean3, sigma3):
    return (slope * x + intercept + scale1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))
            + scale2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2))
            + scale3 * np.exp(-((x - mean3)**2) / (2 * sigma3**2)))


def transmission(x, trans1, scale1, mean1, sigma1, trans2, scale2, mean2, sigma2, trans3, scale3, mean3, sigma3, trans4, scale4, mean4, sigma4):
    return (trans1 * (np.exp(-scale1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))) - 1) +
            trans2 * (np.exp(-scale2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2))) - 1) +
            trans3 * (np.exp(-scale3 * np.exp(-((x - mean3)**2) / (2 * sigma3**2))) - 1) +
            trans4 * (np.exp(-scale4 * np.exp(-((x - mean4)**2) / (2 * sigma4**2))) - 1))


# Define the folder and file paths
folder_name = 'data8'
titles = [f'scope_{i}' for i in range(0, 12)]
data0 = pd.read_csv(f'{folder_name}/clean_data/{titles[0]}_cropped.csv')

# Producing single numpy arrays for manipulation with functions
volt_laser = data0['volt_laser'].to_numpy()
volt_piezo = data0['piezo_fitted'].to_numpy()
timestamp = data0['timestamp'].to_numpy()


param_bounds = ([0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf])

popt_rough, pcov_rough = curve_fit(doppler_envelope_rough, volt_piezo,
                                   volt_laser, bounds=param_bounds, maxfev=10000)
print(popt_rough)


vp = np.linspace(min(volt_piezo), max(volt_piezo), 500)
fit_rough = doppler_envelope_rough(vp, *popt_rough)
line = popt_rough[0] * vp + popt_rough[1]


plt.figure()
plt.scatter(volt_piezo, volt_laser, label='Laser intensity',
            color='blue', s=5, marker='.')
plt.plot(vp, fit_rough, label='fit',
         color='red', linewidth=2, marker='')
plt.plot(vp, line, label='Offset line',
         color='green', linewidth=2, marker='')
plt.xlabel('Volt piezo [V]')
plt.ylabel('Volt Laser [V]')
plt.title(f'Rough fit for {titles[0]}')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'data8/figures/fit_peaks/rough_fit_{titles[0]}.pdf')
plt.close()

just_peaks = volt_laser - doppler_envelope_rough(volt_piezo, *popt_rough)
peaks_indices, _ = find_peaks(
    just_peaks, prominence=(0.003, 0.03), distance=100)

peaks_indices = np.delete(peaks_indices, -1)
peaks_indices = np.delete(peaks_indices, 3)

piezo_peaks = np.array(volt_piezo[peaks_indices])
time_peaks = np.array(timestamp[peaks_indices])
laser_peaks = np.array(volt_laser[peaks_indices])
y_peaks = np.array(just_peaks[peaks_indices])


# plt.figure()
# plt.plot(volt_piezo, just_peaks, label='Peaks without doppler',
#          color='blue', markersize=3, linewidth=2, marker='.')
# plt.scatter(piezo_peaks, y_peaks, label='Peaks',
#             color='red', s=20, marker='x')
# plt.xlabel('Volt Piezo [V]')
# plt.ylabel('Laser peaks [V]')
# plt.title(f'Rediduals {titles[0]}')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'data8/figures/fit_peaks/rough_residuals_{titles[0]}.pdf')
# plt.close()


remove_offset = volt_laser - popt_rough[0] * volt_piezo - popt_rough[1]

p0 = [1, 0.05, piezo_peaks[0], 1.6, 1, 0.05, piezo_peaks[2], 1.6,
      1, 0.02, piezo_peaks[3], 1.6, 1, 0.03, piezo_peaks[4], 1.6]

lower_bounds = [0, 0, -np.inf, 0,
                0, 0, -np.inf, 0,
                0, 0, -np.inf, 0,
                0, 0, -np.inf, 0]

upper_bounds = [np.inf, np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf, np.inf]

popt_off, pcov_off = curve_fit(f=transmission, xdata=volt_piezo,
                               ydata=remove_offset, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
print(popt_off)

fit_off = transmission(vp, *popt_off)

plt.figure()
plt.plot(volt_piezo, remove_offset, label='Data',
         color='blue', markersize=5, marker='.')
plt.plot(vp, fit_off, label='Fit result',
         color='red', markersize=5, marker='')
plt.xlabel('Volt piezo [V]')
plt.ylabel('Laser without offset [V]')
plt.title(f'Removing offset from {titles[0]}')
plt.grid()
plt.legend()
plt.savefig(f'data8/figures/fit_peaks/no_offset_{titles[0]}.pdf')

residuals = remove_offset - transmission(volt_piezo, *popt_off)
lor, cov = fp.fit_peaks_spectroscopy(
    x=volt_piezo, y=residuals, height=0.003, distance=100)

residuals_peaks = residuals[peaks_indices]

x0_list = []
A_list = []
gamma_list = []
off_list = []
dx0 = []
dA = []
dgamma = []
doff = []

for popt_rough, pcov_rough in zip(lor, cov):
    A_list.append(popt_rough[0])
    x0_list.append(popt_rough[1])
    gamma_list.append(popt_rough[2])
    off_list.append(popt_rough[3])
    dA.append(np.sqrt(pcov_rough[0, 0]))
    dx0.append(np.sqrt(pcov_rough[1, 1]))
    dgamma.append(np.sqrt(pcov_rough[2, 2]))
    doff.append(np.sqrt(pcov_rough[3, 3]))

x0_list = np.array(x0_list)
A_list = np.array(A_list)
gamma_list = np.array(gamma_list)
off_list = np.array(off_list)
dx0 = np.array(dx0)
dA = np.array(dA)
dgamma = np.array(dgamma)
doff = np.array(doff)

pf.plot_time_laser_fit(volt_piezo, residuals, f'data8/figures/fit_peaks/residuals_{titles[0]}.pdf',
                       A_list, x0_list, gamma_list, off_list, piezo_peaks, residuals_peaks, save=True)

freq = [377108945610922.8, 377109126401922.8, 377109307192922.8,
        377111224766631.8, 377111633094631.8, 377112041422631.8]

output_file = f'data8/clean_data/{titles[0]}_peaks_fit.csv'
df = pd.DataFrame()
df['indices'] = peaks_indices
df['timestamp'] = time_peaks
df['phtodiode'] = laser_peaks
df['piezo_peaks'] = piezo_peaks
df['freq'] = freq
df['lor_A'] = A_list
df['lor_mean'] = x0_list
df['lor_gamma'] = gamma_list
df['lor_off'] = off_list
df['lor_d_A'] = dA
df['lor_d_mean'] = dx0
df['lor_d_gamma'] = dgamma
df['lor_d_off'] = doff

df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

data0['offset'] = remove_offset
data0.to_csv(f'data8/clean_data/{titles[0]}_offset.csv', index=False)

