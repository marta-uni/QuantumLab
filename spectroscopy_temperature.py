import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import pandas as pd

#load Rb transitions
Rb = [377104390084020.94, 377104798412020.94, 377105206740020.94, 377105909878483.7, 377106090669483.7, 377106271460483.7, 377108945610922.8, 377109126401922.8, 377109307192922.8, 377111224766631.8, 377111633094631.8, 377112041422631.8]
Rb_labels = ['21', 'cross', '22', '32', 'cross', '33', '22', 'cross', '23', '11', 'cross', '12']

#load data + plot
const_i_filename = "data/Fixed_ld00000.csv"
const_i_data = pd.read_csv(const_i_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})
fn.plotting3(const_i_data['timestamp'], const_i_data['photodiode'], const_i_data['volt_piezo']/10, const_i_data['volt_ld'], 'time', 'pd', 'piezo/10', 'ld', 'constant diode current, modulated piezo position', 'figures_temperature/time_constdiode.pdf', save = True)

#load peaks
const_i_peaks_filename = "data/Fixed_ld00000_peaks (1).csv"
const_i_peaks = pd.read_csv(const_i_peaks_filename, skiprows=1, names=['indices','timestamp', 'pd_peaks','piezo_peaks','ld_peaks','freq', 'lor_mean', 'lor_A', 'lor_gamma'])

#fitting peaks data to find freq conversions: dv/dVp (at const diode current) and dv/dt (v = nu)
#if we want to use lorentzian, the one in piezovolt doesnt work anymore
const_i_coeff1, const_i_coeff2 = fn.plot_fits(const_i_peaks['piezo_peaks'], const_i_peaks['freq'], 'piezo_peaks', 'freq', 'Constant diode current, modulated piezo position', "figures_temperature/fit_constdiode.pdf", save=True)
dv_dVp = const_i_coeff1[0]
const_i_freq_volt = dv_dVp * const_i_data['volt_piezo'] + const_i_coeff1[1]
fn.plotting(const_i_freq_volt, const_i_data['photodiode'],'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n constant diode current, modulated piezo position', "figures_temperature/spec_constdiode_volt.pdf", save=True)


t_const_i_coeff1, t_const_i_coeff2 = fn.plot_fits(const_i_peaks['timestamps'], const_i_peaks['freq'], 'time', 'freq', 'Fixed Piezo, modulated diode current: time calib', "figures_temperature/timefit_constdiode.pdf", save=True)
dv_dt = t_const_i_coeff1[0]
const_i_freq_time = dv_dt * const_i_data['timestamp'] + t_const_i_coeff1[1]
fn.plotting(const_i_freq_time, const_i_data['photodiode'],'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n constant diode current, modulated piezo position', "figures_temperature/spec_constdiode_time.pdf", save=True)

#comparison:
fn.plotting(const_i_data['photodiode'], const_i_freq_time - const_i_freq_volt, 'signal at photodiode','calibrated frequency', 'spectroscopy of Rb, \n constant diode current, modulated piezo position', "figures_temperature/spec_constdiode_diff.pdf", save=True)

#conclusion: calibrating with time or with piezo voltage doesnt change anything

#so now:

t_const_i_coeff1, t_const_i_coeff2 = fn.plot_fits(const_i_peaks['lor_mean'], const_i_peaks['freq'], 'time', 'freq', 'Fixed Piezo, modulated diode current: time calib', "figures_temperature/timefit_constdiode.pdf", save=True)
dv_dt = t_const_i_coeff1[0]
const_i_freq_time = dv_dt * const_i_data['timestamp'] + t_const_i_coeff1[1]

