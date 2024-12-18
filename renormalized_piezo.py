import os
import pandas as pd
import plotting_functions as pf
import functions2 as fn2
import numpy as np

c = 3e8
l = 50e-3
fsr_freq = c/(2*l)

'''This code computes the conversion between voltages on the piezo and frequencies, imposing that peaks are evenly
spaced and fitting them to a parabola. This process is followed both in the assumption that the cavity is confocal
(even and odd modes are evenly spaced) and without this assumption (only even modes are considered). It then saves
a csv file with 3 columns: freq_confoc, freq_non_confoc, volt_laser.'''

save_data = False

# Define the folder and file paths
folder_name = 'data6'
file_paths = [
    f"{folder_name}/clean_data/scope_{i}_cropped.csv" for i in range(15, 26)]

# Loop through each file in the file_paths
for file_path in file_paths:
    file_name = os.path.splitext(file_path)[0].replace('_cropped', '')
    figure_path = f"{folder_name}/figures/{os.path.basename(file_name)}"

    '''prepare data (read, crop, fit piezo)'''

    # Read the CSV file, skip the first 2 rows, and specify the data types
    data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo', 'piezo_fitted'], dtype={
                       'timestamp': float, 'volt_laser': float, 'volt_piezo': float, 'piezo_fitted': float})

    volt_laser = data['volt_laser'].to_numpy()
    piezo_fitted = data['piezo_fitted'].to_numpy()

    # ASSUME CONFOCALITY

    print(file_name)
    print('Assuming confocality')

    # Find peaks
    # use this one to have 5 peaks (assume confocality within some precision)
    xpeaks, ypeaks, indices = fn2.peaks_hfsr(piezo_fitted, volt_laser)

    index = next((i for i, x in enumerate(xpeaks) if (
        x >= -8) or (ypeaks[i] > 0.6)), len(xpeaks))
    xpeaks = xpeaks[index:]
    ypeaks = ypeaks[index:]
    indices = indices[index:]

    # Generate expected frequencies
    expected_freq = np.arange(0, fsr_freq/2 * len(xpeaks), fsr_freq/2)

    # find calibration, and plot the data
    coeffs1, coeffs2 = fn2.plot_fits(xpeaks, expected_freq, "Peaks in piezo voltage (V)",
                                     "Expected frequency (Hz)", figure_path + "_calibration_confocal.pdf", confocal=True, save=False)

    # convert piezo voltages into frequencies
    calibrated_freqs = coeffs2[0] * piezo_fitted**2 + \
        coeffs2[1] * piezo_fitted + coeffs2[2]

    pf.plot_calibrated_laser(calibrated_freqs, volt_laser, figure_path +
                             "_data_confocal.pdf", ": assuming confocality", save=False)
    # pf.plot_generic(calibrated_freqs, volt_laser, "calibrated_freqs",
    #                 "volt_laser", file_name + "calibrated_data_confocal")

    peak_freq = calibrated_freqs[indices]

    # displacement of the odd peaks from half the fsr (in Hz)
    displacement1 = (peak_freq[1] - peak_freq[0]) - \
        (peak_freq[2] - peak_freq[0]) / 2
    displacement2 = (peak_freq[3] - peak_freq[2]) - \
        (peak_freq[4] - peak_freq[2]) / 2

    print(
        f'Displacements from confocality are:\t{displacement1/1e6:.0f} MHz\t{displacement2/1e6:.0f} MHz\n')

    # DONT ASSUME CONFOCALITY
    print('Without assuming confocality')

    x_nonconfoc = xpeaks[::2]
    y_nonconfoc = ypeaks[::2]
    i_nonconfoc = indices[::2]

    # generate expected frequencies
    expected_freq = np.arange(0, fsr_freq * len(x_nonconfoc), fsr_freq)

    # find calibration, and plot the data
    coeffs1, coeffs2 = fn2.plot_fits(x_nonconfoc, expected_freq, "Peaks in piezo voltage (V)",
                                     "Expected frequency (Hz)", figure_path + "_calibration_non_confocal.pdf", confocal=False, save=False)

    # convert piezo voltages into frequencies
    calibrated_freqs_1 = coeffs2[0] * piezo_fitted**2 + \
        coeffs2[1] * piezo_fitted + coeffs2[2]

    pf.plot_calibrated_laser(calibrated_freqs_1, volt_laser, figure_path +
                             "_data_non_confocal.pdf", ": without assuming confocality", save=False)
    # pf.plot_generic(calibrated_freqs, volt_laser, "calibrated_freqs", "volt_laser", file_name + "calibrated_data-nonconfocal")

    peak_freq = calibrated_freqs_1[indices]

    # displacement of the odd peaks from half the fsr (in Hz)
    displacement1 = (peak_freq[1] - peak_freq[0]) - \
        (peak_freq[2] - peak_freq[0]) / 2
    displacement2 = (peak_freq[3] - peak_freq[2]) - \
        (peak_freq[4] - peak_freq[2]) / 2

    print(
        f'Displacements from confocality are:\t{displacement1/1e6:.0f} MHz\t{displacement2/1e6:.0f} MHz\n')

    if save_data:
        data = {'freq_confoc': calibrated_freqs,
                'freq_non_confoc': calibrated_freqs_1,
                'volt_laser': volt_laser}
        df = pd.DataFrame(data)

        new_file_path = f"{folder_name}/clean_data/{os.path.basename(file_name)}_calib.csv"
        df.to_csv(new_file_path, index=False)
        print(f"Data saved to {new_file_path}")
