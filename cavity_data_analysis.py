import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
import numpy as np
import os

'''define filename'''
file_name = 'scope_2'
folder_name = 'data1'
file_path = folder_name + '/' + file_name + '.csv'
os.makedirs("figures", exist_ok=True)

'''prepare data (read, crop, fit piezo)'''

# Read the CSV file, skip the first 2 rows, and specify the data types
data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo'],
                   dtype={'timestamp': float, 'volt_laser': float, 'volt_piezo': float})

#remove any rows with NaN values
data_cleaned = data.dropna()

# Extract the columns
timestamps = data_cleaned['timestamp'].to_numpy()  # Convert to NumPy array
volt_laser = data_cleaned['volt_laser'].to_numpy()  # Convert to NumPy array
volt_piezo = data_cleaned['volt_piezo'].to_numpy()  # Convert to NumPy array

# crop data to one piezo cycle
result = fn.crop_to_min_max(timestamps, volt_laser, volt_piezo)

# redefine the arrays with the cropped versions
timestamps = result[0]  
volt_laser = result[1]  
volt_piezo = result[2]

#fit the piezo data
piezo_fitted = fn.fit_piezo_line(timestamps, volt_piezo)

# Plot volt_piezo vs. timestamp  
figure_name = 'data2/figures/' + file_name + "_time.pdf"
fn.plot_voltage_vs_time(timestamps, volt_laser, volt_piezo, piezo_fitted, figure_name)


''' find conversion between piezo volt and freq, generate calibrated x values '''

# calculate expected FSR with parameters
c = 3e8
l = 50e-3 
fsr_freq = c/(2*l)

print('Expected free spectral range =', f"{fsr_freq:.2e}", 'Hz')
expected_wavelength = 780e-9

#find peaks 
xpeaks, ypeaks = fn.peaks(piezo_fitted, volt_laser)

#assume that the distance between the two highest peaks is the fsr
top_two_indices = np.argsort(ypeaks)[-2:]  # Get indices of the two largest values
fsr_volt = np.abs(xpeaks[top_two_indices[0]] - xpeaks[top_two_indices[1]])
print (fsr_volt)

conv_coeff = fsr_freq/fsr_volt 

xvalues_freq = piezo_fitted * conv_coeff + c/expected_wavelength

#assume that the first two peaks give the distance between the modes
#mode_distance = xpeaks[1] - xpeaks[0] # NB this is useless for now


# Plot volt_piezo vs. volt_laser
figure_name = folder_name + '/figures/' + file_name + "_laservolt.pdf"
fn.plot_piezo_laser(piezo_fitted, volt_laser, xpeaks, ypeaks, figure_name)
figure_name = folder_name + '/figures/' + file_name + "_calibrated.pdf"
fn.plot_calibrated_laser(xvalues_freq, volt_laser, figure_name)

