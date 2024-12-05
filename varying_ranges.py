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


data = pd.read_csv('data9/clean_data/Fixed_ld00000_frequencies_cropped_2.csv')

peaks = pd.read_csv('data9/clean_data/Fixed_ld00000_peaks_fit.csv')

frequencies = data['frequencies'].to_numpy()
photodiode = data['offset'].to_numpy()

# mask = (frequencies >= 5e8 + + 3.7711e14) & (frequencies <= 3e9 + 3.7711e14)
good_points = (frequencies >= peaks['freq'][4]) & (frequencies <= 3e9 + 3.7711e14)
restricted_freq = frequencies[good_points]
restricted_pd = photodiode[good_points]

lower_bounds = [0, 0, 0, 273]
upper_bounds = [np.inf, np.inf, np.inf, 500]

temp = []
d_temp = []
V_left = []

print(len(restricted_freq))

while (restricted_freq[0] <= (21.5e8 + 3.7711e14)):
    popt, pcov = curve_fit(transmission_temp_no_off, xdata=restricted_freq,
                           ydata=restricted_pd, bounds=(lower_bounds, upper_bounds), maxfev=10000)
    temp.append(popt[3])
    d_temp.append(np.sqrt(pcov[3, 3]))
    V_left.append(restricted_freq[0])
    restricted_freq = restricted_freq[1:]
    restricted_pd = restricted_pd[1:]

plt.figure()
plt.errorbar(V_left, temp, d_temp, label='Data', color='blue',
             markersize=5, marker='.', linestyle='')
plt.xlabel('Left end of fitting range [Hz]')
plt.ylabel('Temperatures [K]')
plt.title('Varying fitting range')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('data9/figures/temperature/left_end_weird.pdf')

V_left = np.array(V_left)
temp = np.array(temp)
d_temp = np.array(d_temp)

good_points = (d_temp <= 100)
temp = temp[good_points]
d_temp = d_temp[good_points]
V_left = V_left[good_points]

plt.figure()
plt.errorbar(V_left, temp, d_temp, label='Data', color='blue',
             markersize=5, marker='.', linestyle='')
plt.xlabel('Left end of fitting range [Hz]')
plt.ylabel('Temperatures [K]')
plt.title('Varying fitting range')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('data9/figures/temperature/left_end_clean.pdf')
plt.show()