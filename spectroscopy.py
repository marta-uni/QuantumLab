import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import pandas as pd

#load Rb transitions
Rb = [377104390084020.94, 377104798412020.94, 377105206740020.94, 377105909878483.7, 377106090669483.7, 377106271460483.7, 377108945610922.8, 377109126401922.8, 377109307192922.8, 377111224766631.8, 377111633094631.8, 377112041422631.8]
Rb_labels = ['21', 'cross', '22', '32', 'cross', '33', '22', 'cross', '23', '11', 'cross', '12']

#load data
const_piezo_filename = "data/Fixed_piezo00000.csv"
const_i_filename = "data/Fixed_ld00000.csv"
spec_filename = 'data/fullspec00000.csv'

const_piezo_data = pd.read_csv(const_piezo_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})
const_i_data = pd.read_csv(const_i_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})
spec_data = pd.read_csv(spec_filename, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float})

#plotting

fn.plotting3(const_i_data['timestamp'], const_i_data['photodiode'], const_i_data['volt_piezo']/10, const_i_data['volt_ld'], 'time', 'pd', 'piezo/10', 'ld', 'constant diode current, modulated piezo position', 'figures/time_constdiode.pdf', save = True)
fn.plotting3(const_piezo_data['timestamp'], const_piezo_data['photodiode'], const_piezo_data['volt_piezo']/10, const_piezo_data['volt_ld'], 'time', 'pd', 'piezo/10', 'ld', 'constant piezo current, modulated diode', 'figures/time_constpiezo.pdf', save = True)
fn.plotting3(spec_data['timestamp'], spec_data['photodiode'], spec_data['volt_piezo']/10, spec_data['volt_ld'], 'time', 'pd', 'piezo/10', 'ld', 'spectroscopy', 'figures/time_spec.pdf', save = True)

#load peaks
const_i_peaks_filename = "data/Fixed_ld00000_peaks (1).csv"
const_piezo_peaks_filename = "data/Fixed_piezo00000_peaks (1).csv"
spec_peaks_filename = 'data/fullspec00000_peaks (1).csv'

const_piezo_peaks = pd.read_csv(const_piezo_peaks_filename, skiprows=1, names=['indices', 'timestamp','pd_peaks','piezo_peaks','ld_peaks','freq'], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})
const_i_peaks = pd.read_csv(const_i_peaks_filename, skiprows=1, names=['indices','timestamp', 'pd_peaks','piezo_peaks','ld_peaks','freq'], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})
spec_peaks = pd.read_csv(spec_peaks_filename, skiprows=1, names=['indices','timestamp','pd_peaks','piezo_peaks','ld_peaks','freq'], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})

#fitting the data and plotting
const_i_coeff1, const_i_coeff2 = fn.plot_fits(const_i_peaks['ld_peaks'], const_i_peaks['freq'], 'ld_peaks', 'freq', 'Constant diode current, modulated piezo position', "figures/fit_constdiode.pdf", save=True)
const_piezo_coeff1, const_piezo_coeff2 =fn.plot_fits(const_piezo_peaks['piezo_peaks'], const_piezo_peaks['freq'], 'piezo_peaks', 'freq', 'constant piezo position, modulated diode current', "figures/fit_constpiezo.pdf", save=True)

t_const_i_coeff1, t_const_i_coeff2 = fn.plot_fits(const_i_peaks['timestamp'], const_i_peaks['freq'], 'time', 'freq', 'Fixed Piezo, modulated diode current: time calib', "figures/timefit_constdiode.pdf", save=True)
t_const_piezo_coeff1, t_const_piezo_coeff2 = fn.plot_fits(const_piezo_peaks['timestamp'], const_piezo_peaks['freq'], 'time', 'freq', 'Fixed diode current, modulated piezo: time calib', "figures/timefit_constpiezo.pdf", save=True)
t_spec_coeff1, t_spec_coeff2 = fn.plot_fits(spec_peaks['timestamp'], spec_peaks['freq'], 'time', 'freq', 'Full transitions spectroscopy: time calib', "figures/timefit_spec.pdf", save=True)

#frequency conversion

#quadratic:
#const_i_freq = const_i_coeff2[0] * const_i_data['volt_ld']**2 + const_i_coeff2[1] * const_i_data['volt_ld'] + const_i_coeff2[2]
#const_piezo_freq = const_piezo_coeff2[0] * const_piezo_data['volt_piezo']**2 + const_piezo_coeff2[1] * const_piezo_data['volt_piezo'] + const_piezo_coeff2[2]
#spec_freq = t_spec_coeff2[0] * spec_data['timestamp']**2+ t_spec_coeff2[1]* spec_data['timestamp'] + t_spec_coeff2[2] 

#linear:
const_i_freq = const_i_coeff1[0] * const_i_data['volt_ld'] + const_i_coeff1[1]
const_piezo_freq = const_piezo_coeff1[0] * const_piezo_data['volt_piezo']+ const_piezo_coeff1[1]
spec_freq = t_spec_coeff1[0] * spec_data['timestamp']+ t_spec_coeff1[1]

#plot data vs frequencies
fn.plotting(const_i_freq, const_i_data['photodiode'],'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n constant diode current, modulated piezo position', "figures/spec_constdiode.pdf", save=True)
fn.plotting(const_piezo_freq, const_piezo_data['photodiode'], 'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n constant piezo position, modulated diode current', "figures/spec_constpiezo.pdf", save=True)
fn.plotting(spec_freq, spec_data['photodiode'], 'calibrated frequency', 'spec', 'everything modulated', "figures/spec_total_spec.pdf", save=True)


dv_dt_spec = t_spec_coeff1[0]
dv_di = const_piezo_coeff1[0]
di_dt = -9.000777862062634
dv_dp = const_i_coeff1[0]
dp_dt = 268.55520938050694

print ('dv/di', dv_di )
print ('di/dt', di_dt )
print ('dv/dp', dv_dp )
print ('dp/dt', dp_dt )

result = dv_di * di_dt + dv_dp * dp_dt
print(f'dv/dt full spec = {dv_dt_spec:.4e}')
print(f'dv/di * di/dt + dv/dp * dp/dt = {result:.4e}')
