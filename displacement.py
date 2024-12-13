import pandas as pd
import plotting_functions as pf
from scipy.signal import find_peaks
import fit_peaks as fp
import numpy as np
import functions as fn

'''This code tries to estimate how accurate is our assumption of confocality. It does so computing the
displacement of the odd modes (in MHz) from half the FSR (expected to be 0). Then it also computes the
separation between the top odd peak and its closest neighbour (see plots).'''


# putting this here, because it will only be useful in this file
def odd_peaks(x, y, fig=False, figname='odd.pdf', save=False):
    '''
    Function computing distance between close peaks in odd modes.
    If necessary it plots data, with lorentzian fits of peaks.
    '''
    lor, cov = fp.fit_peaks(x, y, height=0.042, distance=80)

    if fig:
        xpeaks, ypeaks, peak_widths = fn.peaks(
            x, y, height=0.042, distance=80)
        A_list = []
        x0_list = []
        gamma_list = []

        for popt in lor:
            A_list.append(popt[0])
            gamma_list.append(popt[2])
            x0_list.append(popt[1])

        pf.plot_freq_laser_fit(x, y, file_name=figname, A=A_list,
                               x0=x0_list, gamma=gamma_list, xpeaks=xpeaks, ypeaks=ypeaks, width=peak_widths, save=save)

    close_peaks = lor[1][1] - lor[0][1]
    d_close = np.sqrt(cov[1][1, 1] + cov[1][1, 1])

    return close_peaks, d_close


for i in range(15, 26):
    file_name = "scope_" + str(i)
    folder_name = "data6/clean_data"
    file_path = folder_name + "/" + file_name + "_calib.csv"
    print(file_name)

    data = pd.read_csv(file_path)

    freq_confoc = data["freq_confoc"].to_numpy()
    freq_non_confoc = data["freq_non_confoc"].to_numpy()
    volt_laser = data["volt_laser"].to_numpy()

    # finding peaks positions in both files
    peak_indices, _ = find_peaks(volt_laser, height=0.08, distance=400)
    index = next((i for i, x in enumerate(peak_indices) if freq_confoc[x] > -5e8),
                 len(peak_indices))
    peak_indices = peak_indices[index:]

    # tem00 will be in positions 0, 2, 4; top odd peaks will be in 1, 3
    peaks_confoc = freq_confoc[peak_indices]
    peaks_non_confoc = freq_non_confoc[peak_indices]

    # computing displacement in fsr from 0 to 3GHz with confocal assumption
    displacement_confoc_1 = peaks_confoc[1] - \
        peaks_confoc[0] - (peaks_confoc[2] - peaks_confoc[0]) / 2
    # computing displacement in fsr from 3 to 6GHz with confocal assumption
    displacement_confoc_2 = peaks_confoc[3] - \
        peaks_confoc[2] - (peaks_confoc[4] - peaks_confoc[2]) / 2
    # computing displacement in fsr from 0 to 3GHz without confocal assumption
    displacement_non_confoc_1 = peaks_non_confoc[1] - peaks_non_confoc[0] - (
        peaks_non_confoc[2] - peaks_non_confoc[0]) / 2
    # computing displacement in fsr from 3 to 6GHz without confocal assumption
    displacement_non_confoc_2 = peaks_non_confoc[3] - peaks_non_confoc[2] - (
        peaks_non_confoc[4] - peaks_non_confoc[2]) / 2

    print("Simple displacement estimation: frequency separation between odd modes and half the fsr.")
    print(
        f"Assuming confocality:\t{displacement_confoc_1/1e6:.0f} MHz\t{displacement_confoc_2/1e6:.0f} MHz")
    print(
        f"Without assuming confocality:\t{displacement_non_confoc_1/1e6:.0f} MHz\t{displacement_non_confoc_2/1e6:.0f} MHz")

    # computing close peaks separation for odd modes in fsr from 0 to 3GHz with confocal assumption
    close_peaks_confoc_1, d_close_confoc_1 = odd_peaks(
        freq_confoc[peak_indices[1] - 600: peak_indices[1] + 200], volt_laser[peak_indices[1] - 600: peak_indices[1] + 200])
    # computing close peaks separation for odd modes in fsr from 3 to 6GHz with confocal assumption
    close_peaks_confoc_2, d_close_confoc_2 = odd_peaks(
        freq_confoc[peak_indices[3] - 600: peak_indices[3] + 200], volt_laser[peak_indices[3] - 600: peak_indices[3] + 200])
    # computing close peaks separation for odd modes in fsr from 0 to 3GHz without confocal assumption
    close_peaks_non_confoc_1, d_close_non_confoc_1 = odd_peaks(
        freq_non_confoc[peak_indices[1] - 600: peak_indices[1] + 200], volt_laser[peak_indices[1] - 600: peak_indices[1] + 200])
    # computing close peaks separation for odd modes in fsr from 3 to 6GHz without confocal assumption
    close_peaks_non_confoc_2, d_close_non_confoc_2 = odd_peaks(
        freq_non_confoc[peak_indices[3] - 600: peak_indices[3] + 200], volt_laser[peak_indices[3] - 600: peak_indices[3] + 200])

    # pf.plot_calibrated_laser(freq_confoc[peak_indices[3] - 600: peak_indices[3] + 200],
    #                          volt_laser[peak_indices[3] - 600: peak_indices[3] + 200], "name")

    print('Close peaks separation in odd modes:')
    print(
        f'Assuming confocality:\t{close_peaks_confoc_1/1e6:.2f} +/- {d_close_confoc_1/1e6:.2f} MHz' +
        f'\t{close_peaks_confoc_2/1e6:.2f} +/- {d_close_confoc_2/1e6:.2f} MHz')
    print(
        f'Without assuming confocality:\t{close_peaks_non_confoc_1/1e6:.2f} +/- {d_close_non_confoc_1/1e6:.2f} MHz' +
        f'\t{close_peaks_non_confoc_2/1e6:.2f} +/- {d_close_non_confoc_2/1e6:.2f} MHz\n')
