import pandas as pd
import functions as fn
import numpy as np

'''define filename'''
file_name = 'scope_2'
folder_name = 'data1'
file_path = folder_name + '/' + file_name + '.csv'

'''prepare data (read, crop, fit piezo)'''

# Read the CSV file, skip the first 2 rows, and specify the data types
data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo'],
                   dtype={'timestamp': float, 'volt_laser': float, 'volt_piezo': float})

# remove any rows with NaN values
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

# fit the piezo data
piezo_fitted = fn.fit_piezo_line(timestamps, volt_piezo)

'''find free spectral range (in voltage units) and finesse from peaks'''

# find peaks and widths
xpeaks, ypeaks, peak_widths = fn.peaks(piezo_fitted, volt_laser)

# combine peaks with corrisponding x values and sort them in descending order
sorted_peaks = sorted(list(zip(xpeaks, ypeaks)),
                      key=lambda x: x[1], reverse=True)

# assume that the distance between the two highest peaks is the fsr
# for now it has to be bigger than 1V
top_peaks = []
for peak in sorted_peaks:
    if (len(top_peaks) >= 2):
        break
    if all(abs(peak[0] - tp[0]) >= 1 for tp in top_peaks):
        top_peaks.append(peak)

fsr_volt = np.abs(top_peaks[0][0] - top_peaks[1][0])
print('fsr in V: ' + str(fsr_volt))

finesses = fsr_volt/peak_widths

print('avg finesse: ' + str(np.mean(finesses)))

'''find length of the cavity from the ratio between fsr
and separation between higher hermite modes'''

# i didn't understand what is the expression for peaks on the side of the TEM00 peaks
# if you have ideas or suggestions please share them
# we should use something like
# mode_distance = xpeaks[1] - xpeaks[0]

''' find conversion between piezo volt and freq, generate calibrated x values '''

# i'm commenting this section as we don't quite know the length of the cavity
# i'm also a bit confused about the accuracy in determination of "expected_wavelength" (the intercept in this conversion)
# it should be done with the wavemeter, but my understanding is that the uncertaninty in this measure will be
# in the hundreds of MHz, while the fsr will be of about ~3GHz

'''
# calculate expected FSR with parameters
c = 3e8
l = 50e-3 # this should come from previous section
fsr_freq = c/(2*l)

print('Expected free spectral range =', f"{fsr_freq:.2e}", 'Hz')
expected_wavelength = 780e-9

conv_coeff = fsr_freq/fsr_volt

xvalues_freq = piezo_fitted * conv_coeff + c/expected_wavelength
'''

'''plotting stuff'''
# Plot volt_piezo vs. timestamp
figure_name = folder_name + '/figures/' + file_name + "_time.pdf"
fn.plot_voltage_vs_time(timestamps, volt_laser,
                        volt_piezo, piezo_fitted, figure_name)
# Plot volt_piezo vs. volt_laser
figure_name = folder_name + '/figures/' + file_name + "_laservolt.pdf"
fn.plot_piezo_laser(piezo_fitted, volt_laser, xpeaks,
                    ypeaks, figure_name, peak_widths)
# figure_name = folder_name + '/figures/' + file_name + "_calibrated.pdf"
# fn.plot_calibrated_laser(xvalues_freq, volt_laser, figure_name)
