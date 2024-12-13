import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import functions as fn

#constants
kb = 1.38e-23  # J/K
rb_mass = 1.67e-27 * 87   # kg
c = 3e8
gamma = 2 * np.pi * 5.75e6
f_12 =  377112041422631.8
f_11 = 377111224766631.8
f_cross = (f_12 + f_11) /2
h = 6.62607015e-34

Isat12 = ( np.pi/3 ) * (h * gamma / c**2) * (f_12)
Isatcross = ( np.pi/3 ) * (h * gamma / c**2) * (f_cross)
Isat11 = ( np.pi/3 ) * (h * gamma / c**2) * (f_11)

area_beam = 1.7671458676442586e-6

#powers = [0.559, 0.238, 0.167, 0.384, 0.384, 0.149, 0.167, 0.123, 0.062, 0.062, 0.068, 0.051]
powers = [0.559, 0.238, 0.167, 0.384, 0.384, 0.149, 0.167, 0.123, 0.062, 0.062]
#powers = [0.559, 0.238, 0.167, 0, 0.384, 0.149, 0.167, 0.123, 0.062, 0.062]
intensities = np.array(powers)/area_beam

save = True
folder_name = 'data_pump/clean_data'
file_names = [f"scope_{i}" for i in range(0, 10)]
file_paths = [f'{folder_name}/{file_name}_cropped.csv' for file_name in file_names]
file_peaks = [f'{folder_name}/{file_name}_peaks_fit.csv' for file_name in file_names]

peaks_11 = []
peaks_cross = [] 
peaks_12 = []

#calculate calibration coeffs from scope 0 

data = pd.read_csv(file_paths[0], skiprows=1, names=['timestamp','photodiode','volt_piezo','volt_ld','offset','frequencies'])
peaks = pd.read_csv(file_peaks[0], skiprows=1, names=['indices','timestamp','pd_peaks','piezo_peaks','freq', 'lor_A','lor_mean','lor_gamma','lor_off','lor_d_A','lor_d_mean','lor_d_gamma','lor_d_off'])

peaks_gamma = np.array(peaks['lor_gamma'])
peaks_mean = np.array(peaks['lor_mean'])
peaks_freqs = np.array(peaks['freq'])

coeff1, __ = fn.plot_fits(peaks_mean, peaks_freqs, 'lormean_piezovolt', 'freq', 'Fixed diode current, modulated piezo position: time calib', f"x.pdf", save= False)

# for each file: reads data, finds calibrated frequencies, plots, and appends peaks to peak arrays
for i, (file_path, file_peak) in enumerate(zip(file_paths, file_peaks)):
    fn.ensure_dir(folder_name + '/figures')
    figure_path = os.path.join(folder_name, 'figures', f"{powers[i]:.3g}_scope_{i}.pdf")

    # Read the CSV file, and specify the data types
    data = pd.read_csv(file_path, skiprows=1, names=['timestamp','photodiode','volt_piezo','volt_ld','offset','frequencies'])
    peaks = pd.read_csv(file_peak, skiprows=1, names=['indices','timestamp','pd_peaks','piezo_peaks','freq', 'lor_A','lor_mean','lor_gamma','lor_off','lor_d_A','lor_d_mean','lor_d_gamma','lor_d_off'])
    
    peaks_gamma = np.array(peaks['lor_gamma'])
    peaks_mean = np.array(peaks['lor_mean'])
    peaks_freqs = np.array(peaks['freq'])
    #peaks_gamma = coeff1[0]*peaks_gamma
    
    frequencies = fn.calibrate(data['volt_piezo'], coeff1)

    fn.plotting(
        frequencies,
        data['photodiode'],
        'freq',
        'Photodiode Signal',
        f'Pump Laser Power = {powers[i]}, Intensity = {intensities[i]}',
        figure_path,
        save
    )
    if i == 0:
        peaks_11.append(peaks_gamma[3])
        peaks_cross.append(peaks_gamma[4])
        peaks_12.append(peaks_gamma[5])
    else:
        peaks_11.append(peaks_gamma[0])
        peaks_cross.append(peaks_gamma[1])
        peaks_12.append(peaks_gamma[2])
    #print(peaks_gamma)

figure_name = os.path.join(folder_name, 'figures', 'linewidths.pdf')
figure_11 = os.path.join(folder_name, 'figures', 'linewidths11.pdf')
figure_cross = os.path.join(folder_name, 'figures', 'linewidthscross.pdf')
figure_12 = os.path.join(folder_name, 'figures', 'linewidths12.pdf')

#remove scope3
intensities= np.delete(intensities, 4)
peaks_11= np.delete(peaks_11, 4)
peaks_cross = np.delete(peaks_cross, 4)
peaks_12= np.delete(peaks_12, 4)

#fn.scatter3(intensities, peaks_11, peaks_cross, peaks_12, 'intensities', 'linewidth_11', 'linewidth_cross', 'linewidth_12', 'linewidth vs pump laser intensity', figure_name, save)
fn.scattering(np.sqrt(1+intensities/Isat11), peaks_11, '(1 + I/Isat)^.5', 'linewidth', 'peak11', figure_11, save)
fn.scattering(np.sqrt(1+intensities/Isatcross), peaks_cross, '(1 + I/Isat)^.5', 'linewidth', 'crosspeak', figure_cross, save)
fn.scattering(np.sqrt(1+intensities/Isat12), peaks_12, '(1 + I/Isat)^.5', 'linewidth', 'peak12', figure_12, save)

print(peaks_11)
print(peaks_12)
