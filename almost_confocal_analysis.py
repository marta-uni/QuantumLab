import pandas as pd
import functions as fn
import fit_peaks as fp
import numpy as np

R = 0.05  # m, mirror radius of curvature
c = 3e8  # speed of light

# just use data6 folder
for i in range(15, 26):
    '''define filename'''
    file_name = 'scope_' + str(i)
    folder_name = 'data6'
    file_path = folder_name + '/' + file_name + '.csv'

    '''prepare data (read, crop, fit piezo)'''

    # Read the CSV file, skip the first 2 rows, and specify the data types
    data = pd.read_csv(file_path, skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo'],
                       dtype={'timestamp': float, 'volt_laser': float, 'volt_piezo': float})

    # remove any rows with NaN values
    data = data.dropna()

    # Extract the columns
    timestamps = data['timestamp'].to_numpy()  # Convert to NumPy array
    # Convert to NumPy array
    volt_laser = data['volt_laser'].to_numpy()
    # Convert to NumPy array
    volt_piezo = data['volt_piezo'].to_numpy()

    # crop data to one piezo cycle
    result = fn.crop_to_min_max(timestamps, volt_laser, volt_piezo)

    # redefine the arrays with the cropped versions
    timestamps = result[0]
    volt_laser = result[1]
    volt_piezo = result[2]

    # fit the piezo data
    piezo_fitted = fn.fit_piezo_line(timestamps, volt_piezo)

    '''find free spectral range (in voltage units) and finesse from peaks'''

    # find peaks and widths with just the find_peak function (only for plotting)
    xpeaks, ypeaks, peak_widths = fn.peaks(piezo_fitted, volt_laser, 0.08, 400)
    # find peaks and widths fitting lorentzians
    lor, cov = fp.fit_peaks(piezo_fitted, volt_laser, 0.08, 400)

    index = next((i for i, x in enumerate(xpeaks) if (
        x >= -8) or (ypeaks[i] > 0.6)), len(xpeaks))
    xpeaks = xpeaks[index:]
    ypeaks = ypeaks[index:]
    peak_widths = peak_widths[index:]
    lor = lor[index:]
    cov = cov[index:]

    x0_list = []
    A_list = []
    gamma_list = []
    dx0 = []
    dA = []
    dgamma = []

    for popt, pcov in zip(lor, cov):
        A_list.append(popt[0])
        x0_list.append(popt[1])
        gamma_list.append(popt[2])
        dA.append(np.sqrt(pcov[0, 0]))
        dx0.append(np.sqrt(pcov[1, 1]))
        dgamma.append(np.sqrt(pcov[2, 2]))

    x0_list = np.array(x0_list)
    A_list = np.array(A_list)
    gamma_list = np.array(gamma_list)
    dx0 = np.array(dx0)
    dA = np.array(dA)
    dgamma = np.array(dgamma)

    fsr_volt, d_fsr_volt = fn.fsr(x0_list, A_list, dx0)
    finesse_list = fsr_volt/gamma_list
    d_finesse_list = finesse_list * \
        np.sqrt((d_fsr_volt/fsr_volt) ** 2 + (dgamma/gamma_list) ** 2)

    finesse, dfinesse = fn.weighted_avg(finesse_list, d_finesse_list)

    if fsr_volt is None:
        print(file_path + ' didn\'t produce a fsr')
        continue

    print('FSR for ' + file_path + f': {fsr_volt:.6f} +/- {d_fsr_volt:.6f} V')
    print(f'Finesse: {finesse:.0f} +/- {dfinesse:.0f}')

    '''find length of the cavity from the ratio between fsr
    and separation between higher hermite modes'''

    expected_mode_distance = 5  # V, adjust this as needed

    # finding principal peaks of even and odd modes, these will be separated by
    # mode_distance_1 / fsr = (1/pi) * arccos(1-L/R)
    # mode_distance_2 / fsr = 1 - (1/pi) * arccos(1-L/R)
    mode_distance_list_1, mode_distance_list_2, md_err_1, md_err_2, = fn.even_odd_modes(
        x0_list, A_list, dx0, expected_mode_distance)

    mode_distance_list_1 = np.array(mode_distance_list_1)
    mode_distance_list_2 = np.array(mode_distance_list_2)
    md_err_1 = np.array(md_err_1)
    md_err_2 = np.array(md_err_2)

    print(
        f'Mode distances:\n{mode_distance_list_1}\t+/-\t{md_err_1}\t(even->odd)')
    print(
        f'{mode_distance_list_2}\t+/-\t{md_err_2}\t(odd->even)')

    L_list_1 = R * (1 - np.cos(np.pi * mode_distance_list_1 / fsr_volt))
    L_list_2 = R * (1 + np.cos(np.pi * mode_distance_list_2 / fsr_volt))

    dL1_dMD = R * np.sin(np.pi * mode_distance_list_1 /
                         fsr_volt) * np.pi / fsr_volt
    dL1_dFSR = dL1_dMD * (-mode_distance_list_1 / fsr_volt)
    dL1_list = np.sqrt((dL1_dMD * md_err_1) ** 2 +
                       (dL1_dFSR * d_fsr_volt) ** 2)

    dL2_dMD = -R * np.sin(np.pi * mode_distance_list_2 /
                          fsr_volt) * np.pi / fsr_volt
    dL2_dFSR = dL2_dMD * (-mode_distance_list_2 / fsr_volt)
    dL2_list = np.sqrt((dL2_dMD * md_err_2) ** 2 +
                       (dL2_dFSR * d_fsr_volt) ** 2)

    L_list = np.concatenate((L_list_1, L_list_2))
    dL_list = np.concatenate((dL1_list, dL2_list))

    L, dL = fn.weighted_avg(L_list, dL_list)

    print(f'L = {L:.6f} +/- {dL:.6f} m')

    ''' find conversion between piezo volt and freq, generate calibrated x values '''

    # i'm a bit confused about the accuracy in determination of "expected_wavelength" (the intercept in this conversion)
    # it should be done with the wavemeter, but my understanding is that the uncertaninty in this measure will be
    # in the hundreds of MHz, while the fsr will be of about ~3GHz

    # calculate expected FSR with parameters
    fsr_freq = c/(2*L)
    d_fsr_freq = fsr_freq * dL / L

    print('FSR for ' + file_path +
          f': {(fsr_freq/1e9):.6f} +/- {(d_fsr_freq/1e9):.6f} GHz (the left one)')

    expected_wavelength = 780e-9

    conv_coeff = fsr_freq/fsr_volt

    displacement_from_half_1 = mode_distance_list_1[0] - (mode_distance_list_1[0] + mode_distance_list_2[0]) / 2
    displacement_from_half_2 = mode_distance_list_1[1] - (mode_distance_list_1[1] + mode_distance_list_2[1]) / 2

    print('Displacement of odd peaks from half the FSR:')
    print(f'First odd peak: {displacement_from_half_1*conv_coeff/1e6:.0f} MHz')
    print(f'Second odd peak: {displacement_from_half_2*conv_coeff/1e6:.0f} MHz\n')

    xvalues_freq = piezo_fitted * conv_coeff + c/expected_wavelength

    # figure_name = folder_name + '/figures/' + file_name + "_time.pdf"
    # fn.plot_voltage_vs_time(timestamps, volt_laser,
    #                         volt_piezo, piezo_fitted, figure_name)

    # figure_name = folder_name + '/figures/' + file_name + "_laservolt_fit.pdf"
    # fn.plot_piezo_laser_fit(piezo_fitted, volt_laser, file_name=figure_name, A=A_list,
    #                         x0=x0_list, gamma=gamma_list, xpeaks=xpeaks, ypeaks=ypeaks, width=peak_widths)

    # figure_name = folder_name + '/figures/' + file_name + "_calibrated.pdf"
    # fn.plot_calibrated_laser(xvalues_freq, volt_laser, figure_name)
