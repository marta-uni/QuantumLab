import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import pandas as pd

'''

this code takes in files scope_{i}_cropped.csv and scope_{i}_peaks_fit.csv, calibrates, generates scope{i}_frequencies.csv with a new column with the calibrated frequencies

'''


'''
#load Rb transitions
Rb = [377104390084020.94, 377104798412020.94, 377105206740020.94, 377105909878483.7, 377106090669483.7, 377106271460483.7, 377108945610922.8, 377109126401922.8, 377109307192922.8, 377111224766631.8, 377111633094631.8, 377112041422631.8]
Rb_labels = ['21', 'cross', '22', '32', 'cross', '33', '22', 'cross', '23', '11', 'cross', '12']
Rb_bools = [False, False, False, False, False, False, False, False, False, False, True, True ]
'''

#code parameters
save = True
folder_name = 'data_pump/clean_data'
file_names = [f"scope_{i}" for i in range(0, 11)]
file_paths = [f'{folder_name}/{file_name}_cropped.csv' for file_name in file_names]
file_peaks = [f'{folder_name}/{file_name}_peaks_fit.csv' for file_name in file_names]

for file_path, file_name, file_peaks in zip(file_paths, file_names, file_peaks):
    #load data from files
    data = pd.read_csv(file_path, skiprows=1, names=['timestamp', 'photodiode', 'volt_piezo', 'volt_ld', 'offset'], dtype={'timestamp': float, 'photodiode': float, 'volt_piezo': float, 'volt_ld': float, 'offset': float})
    peaks = pd.read_csv(file_peaks, skiprows=1, names=['indices','timestamp','pd_peaks','piezo_peaks','freq', 'lor_A','lor_mean','lor_gamma','lor_off','lor_d_A','lor_d_mean','lor_d_gamma','lor_d_off'
], dtype={'indices': int, 'pd_peaks': float, 'piezo_peaks': float, 'ld_peaks': float})

    #turn peaks data into np arrays
    peaks_time = np.array(peaks['timestamp'])
    peaks_mean = np.array(peaks['lor_mean'])
    #peaks_mean_piezo = np.array(peaks['lor_d_mean'])
    peaks_gamma = np.array(peaks['lor_gamma'])
    peaks_freqs = np.array(peaks['freq'])

    '''
    coeff1, __ = fn.plot_fits(peaks_mean_time, peaks_freqs, 'lormean_time', 'freq', 'Fixed diode current, modulated piezo position: time calib', f"{folder_name}/figures/{file_name}_timecalib.pdf", save)
    frequencies = coeff1[0] * data['timestamp'] + coeff1[1]
    frequencies = np.array(frequencies)
    fn.plotting(frequencies, data['photodiode'],'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n constant diode current, modulated piezo position', f"{folder_name}/figures/{file_name}_timecalib_spec.pdf", save)
    '''

    #calculate calibration from piezo peaks 
    coeff1, __ = fn.plot_fits(peaks_mean, peaks_freqs, 'lormean_piezovolt', 'freq', 'Fixed diode current, modulated piezo position: time calib', f"{folder_name}/figures/{file_name}_piezocalib.pdf", save)
    frequencies = coeff1[0] * data['volt_piezo'] + coeff1[1]
    frequencies = np.array(frequencies)
    fn.plotting(frequencies, data['photodiode'],'calibrated frequency', 'signal at photodiode', 'spectroscopy of Rb, \n constant diode current, modulated piezo position', f"{folder_name}/figures/{file_name}_piezocalib_spec.pdf", save)

    #double check frequencies same length as data and then add frequencies to data pandas array. saves to csv
    if len(data) != len(frequencies):
        raise ValueError("Data and frequencies must have the same length.")
    data['frequencies']= frequencies
    data.to_csv(f'{folder_name}/{file_name}_frequencies.csv', index=False)

