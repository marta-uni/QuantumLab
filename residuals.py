import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
import numpy as np
from scipy.optimize import curve_fit
import fit_peaks as fp
from scipy.signal import find_peaks
import plotting_functions as pf


# Define the folder and file paths
folder_name = 'data9'
title = 'Fixed_ld00000'
data = pd.read_csv(f'{folder_name}/clean_data/{title}.csv')
peaks = pd.read_csv(f'{folder_name}/clean_data/{title}_peaks.csv')

# Producing single numpy arrays for manipulation with functions
timestamps = data['timestamp'].to_numpy()
volt_laser = data['volt_laser'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()

time_peaks = peaks['timestamp'].to_numpy()


def doppler_envelope_rough(x, slope, intercept, scale1, mean1, sigma1, scale2, mean2, sigma2, scale3, mean3, sigma3):
    return (slope * x + intercept + scale1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))
            + scale2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2))
            + scale3 * np.exp(-((x - mean3)**2) / (2 * sigma3**2)))


p0 = [-3, 3, -0.7, time_peaks[1], 2.5e-3, -0.2,
      time_peaks[3], 2.5e-3, -0.5, time_peaks[4], 2.5e-3]

param_bounds = ([-np.inf, -np.inf, -np.inf, -0.015, 0, -np.inf, 0.0005, 0, -np.inf, 0.006, 0],
                [0, np.inf, 0, -0.008, np.inf, 0, 0.005, np.inf, 0, 1.27, np.inf])

popt_rough, pcov_rough = curve_fit(
    doppler_envelope_rough, xdata=timestamps, ydata=volt_laser, bounds=param_bounds, p0=p0, maxfev=10000)
print(popt_rough)

t = np.linspace(min(timestamps), max(timestamps), 500)
fit = doppler_envelope_rough(t, *popt_rough)

plt.figure()
plt.plot(timestamps, volt_laser, label='Laser intensity',
         color='blue', markersize=5, marker='.')
plt.plot(t, fit, label='Fit result',
         color='red', markersize=5, marker='')
plt.xlabel('Timestamp [s]')
plt.ylabel('Volt Laser [V]')
plt.title(f'Fitting {title}')
plt.grid()
plt.savefig(f'data9/figures/fit_peaks/rough_fit_{title}.pdf')
plt.legend()

remove_offset = volt_laser - popt_rough[0] * timestamps - popt_rough[1]


def transmission(x, trans1, scale1, mean1, sigma1, trans2, scale2, mean2, sigma2, trans3, scale3, mean3, sigma3, trans4, scale4, mean4, sigma4):
    return (trans1 * np.exp(-scale1 * np.exp(-((x - mean1)**2) / (2 * sigma1**2))) +
            trans2 * np.exp(-scale2 * np.exp(-((x - mean2)**2) / (2 * sigma2**2))) +
            trans3 * np.exp(-scale3 * np.exp(-((x - mean3)**2) / (2 * sigma3**2))) +
            trans4 * np.exp(-scale4 * np.exp(-((x - mean4)**2) / (2 * sigma4**2))))


p0 = [1, 0.5, time_peaks[0], 2.5e-3, 1, 0.5, time_peaks[2], 2.5e-3,
      1, 0.2, time_peaks[3], 2.5e-3, 1, 0.5, time_peaks[4], 2.5e-3]

# not using these
'''lower_bounds = [0, 0, -0.01468, 0,  # first transition
                0, 0, -0.01170, 0,  # second transition
                0, 0, 0.002279, 0,  # third transition
                0, 0, 0.007642, 0]  # fourth transition

upper_bounds = [np.inf, np.inf, -0.01271, np.inf,  # first transition
                np.inf, np.inf, -0.01017, np.inf,  # second transition
                np.inf, np.inf, 0.003253, np.inf,  # third transition
                np.inf, np.inf, 0.008765, np.inf]  # fourth transition'''

popt, pcov = curve_fit(transmission, xdata=timestamps,
                       ydata=remove_offset, p0=p0, maxfev=10000)
print(popt)

t_off = np.linspace(min(timestamps), max(timestamps), 500)
fit_off = transmission(t, *popt)

plt.figure()
plt.plot(timestamps, remove_offset, label='Data',
         color='blue', markersize=5, marker='.')
plt.plot(t_off, fit_off, label='Fit result',
         color='red', markersize=5, marker='')
plt.xlabel('Timestamp [s]')
plt.ylabel('Laser without offset [V]')
plt.title(f'Removing offset from {title}')
plt.grid()
plt.savefig(f'data9/figures/fit_peaks/no_offset_{title}.pdf')
plt.legend()

residuals = remove_offset - transmission(timestamps, *popt)
lor, cov = fp.fit_peaks_spectroscopy(
    timestamps, residuals, height=0.016, distance=100)
pk, _ = find_peaks(residuals, height=0.016, distance=100)

lor.pop(3)
cov.pop(3)
pk = np.delete(pk, 3)

xpeaks = timestamps[pk]
ypeaks = residuals[pk]

x0_list = []
A_list = []
gamma_list = []
off_list = []
dx0 = []
dA = []
dgamma = []
doff = []

for popt, pcov in zip(lor, cov):
    A_list.append(popt[0])
    x0_list.append(popt[1])
    gamma_list.append(popt[2])
    off_list.append(popt[3])
    dA.append(np.sqrt(pcov[0, 0]))
    dx0.append(np.sqrt(pcov[1, 1]))
    dgamma.append(np.sqrt(pcov[2, 2]))
    doff.append(np.sqrt(pcov[3, 3]))

x0_list = np.array(x0_list)
A_list = np.array(A_list)
gamma_list = np.array(gamma_list)
off_list = np.array(off_list)
dx0 = np.array(dx0)
dA = np.array(dA)
dgamma = np.array(dgamma)
doff = np.array(doff)

pf.plot_time_laser_fit(timestamps, residuals, f'data9/figures/fit_peaks/residuals_{title}.pdf',
                       A_list, x0_list, gamma_list, off_list, xpeaks, ypeaks, save=True)

output_file = f'data9/clean_data/{title}_peaks_fit.csv'
peaks['lor_A'] = A_list
peaks['lor_mean'] = x0_list
peaks['lor_gamma'] = gamma_list
peaks['lor_off'] = off_list
peaks['lor_d_A'] = dA
peaks['lor_d_mean'] = dx0
peaks['lor_d_gamma'] = dgamma
peaks['lor_d_off'] = doff

peaks.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
