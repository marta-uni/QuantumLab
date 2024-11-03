import pandas as pd
import functions as fn
import numpy as np

# boolean determining wheter we want a plot every time
plot = False

# computing finesses for each file and appending them to this list
finesse_list = []

# computing fsr in volt for each file and appending them to this list
fsr_volt_list = []

# number between 0 and 0.5 determining which part of the piezo voltages
# is not reliable, if there are peaks in the extremes we neglect the file
# e.g bound = 0 will allow every file, bound = 0.1 will skip files with peaks
# in the top (or bottom) 10% of the piezo run
bound = 0.1

# in order to keep consistency it is better to loop on data folders separately
# since i'm lazy i'll just write here the correct indices to loop over for each data
# folder, feel free to improve this solution
# data1 -> range(5)
# data1 -> range(10)
# data3 -> range(10, 20)
for i in range(10, 20):
    '''define filename'''
    file_name = 'scope_' + str(i)
    folder_name = 'data3'
    file_path = folder_name + '/' + file_name + '.csv'

    '''prepare data (read, crop, fit piezo)'''

    # Read the CSV file, skip the first 2 rows, and specify the data types
    data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo'],
                       dtype={'timestamp': float, 'volt_laser': float, 'volt_piezo': float})

    # remove any rows with NaN values
    data_cleaned = data.dropna()

    # Extract the columns
    timestamps = data_cleaned['timestamp'].to_numpy()  # Convert to NumPy array
    # Convert to NumPy array
    volt_laser = data_cleaned['volt_laser'].to_numpy()
    # Convert to NumPy array
    volt_piezo = data_cleaned['volt_piezo'].to_numpy()

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

    if plot:
        # Plot volt_piezo vs. timestamp
        figure_name = folder_name + '/figures/' + file_name + "_time.pdf"
        fn.plot_voltage_vs_time(timestamps, volt_laser,
                                volt_piezo, piezo_fitted, figure_name)
        # Plot volt_piezo vs. volt_laser
        figure_name = folder_name + '/figures/' + file_name + "_laservolt.pdf"
        fn.plot_piezo_laser(piezo_fitted, volt_laser, xpeaks,
                            ypeaks, figure_name, peak_widths)

    valid_range_condition = (min(xpeaks) < bound * (piezo_fitted[-1] - piezo_fitted[0]) + piezo_fitted[0]) or (
        max(xpeaks) > (1 - bound) * (piezo_fitted[-1] - piezo_fitted[0]) + piezo_fitted[0])

    if (valid_range_condition):
        print('peaks out of valid range in ' + file_path)
        continue

    fsr_volt = fn.fsr(xpeaks, ypeaks)

    if fsr_volt is None:
        print(file_path + ' didn\'t produce a fsr')
        continue

    fsr_volt_list.append(fsr_volt)
    finesses = fsr_volt/peak_widths
    finesse_list.extend(finesses)

    '''find length of the cavity from the ratio between fsr
    and separation between higher hermite modes'''

    # i didn't understand what is the expression for peaks on the side of the TEM00 peaks
    # if you have ideas or suggestions please share them
    # we should use something like
    # mode_distance = xpeaks[1] - xpeaks[0]

    ''' find conversion between piezo volt and freq, generate calibrated x values '''

    # i'm commenting out this section as we don't quite know the length of the cavity
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
    if plot:
        figure_name = folder_name + '/figures/' + file_name + "_calibrated.pdf"
        fn.plot_calibrated_laser(xvalues_freq, volt_laser, figure_name)
    '''

print('Finesse: ' + str(np.mean(finesse_list)) +
      ' +/- ' + str(np.std(finesse_list)))
print('Fsr in V: ' + str(np.mean(fsr_volt_list)) +
      ' +/- ' + str(np.std(fsr_volt_list)))
