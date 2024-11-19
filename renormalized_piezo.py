import os
import pandas as pd
import functions2 as fn
import numpy as np

c = 3e8
l = 50e-3
fsr_freq = c/(2*l)
expected_wavelength = 780e-9

'this code reads the csv files, crops them to a window where the piezo voltage is just increasing, removes NaNs and fits the piezo data linearly. It then saves a csv file with  4 columns: timestamps, volt_laser, volt_piezo, piezo_fitted'

# Define the folder and file paths
folder_name = 'data6'
file_paths = [f"{folder_name}/figures/scope_{i}_cropped.csv" for i in range(15, 26)]  # Example: process files scope_24.csv to scope_25.csv

# Loop through each file in the file_paths
for file_path in file_paths:
    file_name = os.path.splitext(file_path)[0].replace('_cropped', '')

    '''prepare data (read, crop, fit piezo)'''

    # Read the CSV file, skip the first 2 rows, and specify the data types
    data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo', 'piezo_fitted'], dtype={'timestamp': float, 'volt_laser': float, 'volt_piezo': float, 'piezo_fitted': float})

    timestamps = data['timestamp'].to_numpy()
    volt_laser = data['volt_laser'].to_numpy()
    volt_piezo = data['volt_piezo'].to_numpy()
    piezo_fitted = data['piezo_fitted'].to_numpy()
    print (timestamps.shape)
    
    #ASSUME CONFOCALITY
    
    # Find peaks 
    xpeaks, ypeaks, indices = fn.peaks_hfsr(piezo_fitted, volt_laser) #use this one to have 5 peaks (assume confocality within some precision)
    
    #generate expected frequencies
    expected_freq = np.arange(0, fsr_freq * len(xpeaks), fsr_freq)

    #find calibration, and plot the data
    coeffs1, coeffs2 = fn.plot_fits(xpeaks, expected_freq, "piezo_voltage_peaks", "expected_frequency", file_name + "calibration-confocal")

    #convert piezo voltages into frequencies
    calibrated_freqs = coeffs2[0] * piezo_fitted**2 + coeffs2[1] *piezo_fitted + coeffs2[2]

    fn.plot_generic(calibrated_freqs, volt_laser, "calibrated_freqs", "volt_laser", file_name + "calibrated data-confocal")


    #generate expected frequencies
    expected_freq = np.arange(0, fsr_freq * len(xpeaks), fsr_freq)

    #find calibration, and plot the data
    coeffs1, coeffs2 = fn.plot_fits(xpeaks, expected_freq, "piezo_voltage_peaks", "expected_frequency", file_name + "calibration-confocal")

    #convert piezo voltages into frequencies
    calibrated_freqs = coeffs2[0] * piezo_fitted**2 + coeffs2[1] *piezo_fitted + coeffs2[2]

    fn.plot_generic(calibrated_freqs, volt_laser, "calibrated_freqs", "volt_laser", file_name + "calibrated data-confocal")
    
    '''
    #DONT ASSUME CONFOCALITY
    
    #Find peaks
    xpeaks, ypeaks, indices = fn.peaks_fsr(piezo_fitted, volt_laser)  #use this one to have 3 peaks
    #generate expected frequencies
    expected_freq = np.arange(0, fsr_freq * len(xpeaks), fsr_freq)

    #find calibration, and plot the data
    coeffs1, coeffs2 = fn.plot_fits(xpeaks, expected_freq, "piezo_voltage_peaks", "expected_frequency", file_name + "calibration-confocal")

    #convert piezo voltages into frequencies
    calibrated_freqs = coeffs2[0] * piezo_fitted**2 + coeffs2[1] *piezo_fitted + coeffs2[2]

    fn.plot_generic(calibrated_freqs, volt_laser, "calibrated_freqs", "volt_laser", file_name + "calibrated data-confocal")


    #generate expected frequencies
    expected_freq = np.arange(0, fsr_freq * len(xpeaks), fsr_freq)

    #find calibration, and plot the data
    coeffs1, coeffs2 = fn.plot_fits(xpeaks, expected_freq, "piezo_voltage_peaks", "expected_frequency", file_name + "calibration-notconfocal")

    #convert piezo voltages into frequencies
    calibrated_freqs = coeffs2[0] * piezo_fitted**2 + coeffs2[1] *piezo_fitted + coeffs2[2]

    fn.plot_generic(calibrated_freqs, volt_laser, "calibrated_freqs", "volt_laser", file_name + "calibrated data-notconfocal") 
    '''
