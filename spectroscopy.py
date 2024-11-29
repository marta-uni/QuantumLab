import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import pandas as pd

#load Rb transitions
Rb = [377104390084020.94, 377104798412020.94, 377105206740020.94, 377105909878483.7, 377106090669483.7, 377106271460483.7, 377108945610922.8, 377109126401922.8, 377109307192922.8, 377111224766631.8, 377111633094631.8, 377112041422631.8]
Rb_labels = ['21', 'cross', '22', '32', 'cross', '33', '22', 'cross', '23', '11', 'cross', '12']

#load data
f_diode_filename = r"data\Fixed_ld00000.csv"
f_diode_data = pd.read_csv(f_diode_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})

f_piezo_filename = r"data\Fixed_piezo00000.csv"
f_piezo_data = pd.read_csv(f_piezo_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})

f_piezo2_filename = r"data\Fixed_piezo00001.csv"
f_piezo2_data = pd.read_csv(f_piezo2_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})

#plotting

fn.plotting3(f_piezo_data['timestamp'], f_piezo_data['photodiode'], f_piezo_data['volt_piezo']/10, f_piezo_data['volt_ld']/10, 'time', 'pd', 'piezo/10', 'ld/10', 'modulated diode current, fixed piezo', r'figures\time_fixed_piezo.pdf', save = True)
fn.plotting3(f_piezo2_data['timestamp'], f_piezo2_data['photodiode'], f_piezo2_data['volt_piezo']/10, f_piezo2_data['volt_ld']/10, 'time', 'pd', 'piezo/10', 'ld/10', 'modulated diode current, fixed piezo', r'figures\time_fixed_piezo2.pdf', save = True)
fn.plotting3(f_diode_data['timestamp'], f_diode_data['photodiode'], f_diode_data['volt_piezo']/10, f_diode_data['volt_ld']/10, 'time', 'pd', 'piezo/10', 'ld/10', 'modulated piezo current, fixed diode', r'figures\time_fixed_diode.pdf', save = True)
#load peaks
f_diode_peaks_filename = r"data\Fixed_ld00000_peaks.csv"
f_piezo_peaks_filename = r"data\Fixed_piezo00000_peaks.csv"
f_piezo2_peaks_filename = r"data\Fixed_piezo00001_peaks.csv"

f_diode_peaks = pd.read_csv(f_diode_peaks_filename, skiprows=1, names=['indices','pd_peaks','piezo_peaks','ld_peaks','freq'], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})
f_piezo_peaks = pd.read_csv(f_piezo_peaks_filename, skiprows=1, names=['indices','pd_peaks','piezo_peaks','ld_peaks','freq'], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})
f_piezo2_peaks = pd.read_csv(f_piezo2_peaks_filename, skiprows=1, names=['indices','pd_peaks','piezo_peaks','ld_peaks','freq'], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})

#fitting the data and plotting
__, f_piezo_coeff2 = fn.plot_fits(f_piezo_peaks['ld_peaks'], f_piezo_peaks['freq'], 'ld_peaks', 'freq', 'Fixed Piezo, modulated diode current', r"figures\fixedpiezofit.pdf", save=True)
#f_piezo2_coeff1, f_piezo2_coeff2 =fn.plot_fits(f_piezo2_peaks['ld_peaks'], f_piezo2_peaks['freq'], 'ld_peaks', 'freq', 'Fixed Piezo, modulated diode current', r"figures\fixedpiezo2.pdf", save=True)
__, f_diode_coeff2 =fn.plot_fits(f_diode_peaks['piezo_peaks'], f_diode_peaks['freq'], 'piezo_peaks', 'freq', 'Fixed diode current, modulated piezo', r"figures\fixeddiodefit.pdf", save=True)

#frequency conversion
f_piezo_freq = f_piezo_coeff2[0] * f_piezo_data['volt_ld']**2 + f_piezo_coeff2[1] * f_piezo_data['volt_ld'] + f_piezo_coeff2[2]
f_diode_freq = f_diode_coeff2[0] * f_diode_data['volt_piezo']**2 + f_diode_coeff2[1] * f_diode_data['volt_piezo'] + f_diode_coeff2[2]

#plot data vs frequencies
fn.plotting(f_piezo_freq, f_piezo_data['photodiode'],'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n modulated diode current, fixed piezo position', r"figures\fixedpiezo_spec.pdf", save=True)
fn.plotting(f_diode_freq, f_diode_data['photodiode'], 'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n fixed diode current, modulated piezo position', r"figures\fixeddiode_spec.pdf", save=True)