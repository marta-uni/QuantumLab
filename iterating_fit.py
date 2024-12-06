import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

mask = (frequencies >= peaks['freq'][4]) & (frequencies <= 3e9 + 3.7711e14)
restricted_freq = frequencies[mask]
restricted_pd = photodiode[mask]

lower_bounds = [0, 0, 0, 273]
upper_bounds = [np.inf, np.inf, np.inf, 500]

temp = []
d_temp = []

popt, pcov = curve_fit(transmission_temp_no_off, xdata=restricted_freq,
                       ydata=restricted_pd, bounds=(lower_bounds, upper_bounds), maxfev=10000)
temp.append(popt[3])
d_temp.append(np.sqrt(pcov[3, 3]))

lower_bounds = [0, 0, 273]
upper_bounds = [np.inf, np.inf, 500]

for i in range(0, 1000):
    def partial_fit(x, a, s, t):
        return transmission_temp_no_off(x=x, ampl=a, scale1=popt[1], scale2=s, temp=t)

    popt, pcov = curve_fit(partial_fit, xdata=restricted_freq,
                           ydata=restricted_pd, bounds=(lower_bounds, upper_bounds), maxfev=10000)
    print(f'{popt[2]}\t{np.sqrt(pcov[2,2])}')
    print(f'{popt[1]}\t{np.sqrt(pcov[1,1])}\n')
    temp.append(popt[2])
    d_temp.append(np.sqrt(pcov[2, 2]))

    def partial_fit_2(x, a, s, t):
        return transmission_temp_no_off(x=x, ampl=a, scale1=s, scale2=popt[1], temp=t)

    p0 = popt
    popt, pcov = curve_fit(partial_fit_2, xdata=restricted_freq,
                           ydata=restricted_pd, bounds=(lower_bounds, upper_bounds), maxfev=10000)
    print(f'{popt[2]}\t{np.sqrt(pcov[2,2])}')
    print(f'{popt[1]}\t{np.sqrt(pcov[1,1])}\n')
    temp.append(popt[2])
    d_temp.append(np.sqrt(pcov[2, 2]))

indices = np.arange(0, len(temp))

plt.figure()
plt.scatter(indices, temp, label='Temperature estimate', color='blue',
            s=5, marker='.')
plt.xlabel('Iterations')
plt.ylabel('Temperatures [K]')
plt.title('Iterative approach')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('data9/figures/temperature/iterations.pdf')
plt.show()
