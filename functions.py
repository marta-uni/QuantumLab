import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

def crop_to_min_max(time, laser_voltage, piezo_voltage):
    '''Crops time and voltage readings in order to include just one
     frequency sweep from the piezo.'''

    if len(piezo_voltage) == 0:
        return None  # Return None if piezo_voltage is empty

    # Find indices of minimum and maximum values
    # Assuming there will be just one full sweep in the dataset
    min_index = np.argmin(piezo_voltage)
    max_index = np.argmax(piezo_voltage)

    # Ensure the possibility to analyse "backwards" sweeps too
    start_index = min(min_index, max_index)
    end_index = max(min_index, max_index)

    # Extract the data between min_index and max_index
    time_window = time[start_index:end_index + 1]
    laser_window = laser_voltage[start_index:end_index + 1]
    piezo_window = piezo_voltage[start_index:end_index + 1]

    return time_window, laser_window, piezo_window


def fit_piezo_line(time, piezo_voltage):
    '''Converts timestamps in voltages on piezo.
    
    Returns voltages from a linear interpolation of input data.'''

    if len(time) == 0 or len(piezo_voltage) == 0 or len(time) != len(piezo_voltage):
        return None  # Return None if the input arrays are empty or of different lengths

    # Fit a line (degree 1 polynomial) to the piezo voltage data
    slope, intercept = np.polyfit(time, piezo_voltage, 1)
    piezo_fit = slope * time + intercept

    return piezo_fit

def peaks(piezo_voltage, laser_voltage):
    '''
    Finds peaks in readings from the photodiode.
    
    Parameters:
    
    piezo_voltage: voltages on the piezo, should be the cleaned version (the output of fit_piezo_line)
    laser_voltage: voltages on the photodiode

    Returns: 

    peaks_xvalues: voltage values on the piezo corresponding to detected peaks
    peaks: detected peaks
    scaled_widths: width of each peak in Volts
    '''
    
    peaks_indices, _ = find_peaks(laser_voltage, height= 0.3, distance=10)
    peaks = laser_voltage[peaks_indices]
    peaks_xvalues = piezo_voltage[peaks_indices]

    widths = peak_widths(laser_voltage, peaks_indices, rel_height=0.5)
    piezo_voltage_spacing = np.mean(np.diff(piezo_voltage))
    scaled_widths = widths[0]*piezo_voltage_spacing

    return peaks_xvalues, peaks, scaled_widths

def fsr(xpeaks, ypeaks):
    '''
    Computes FSR (in volts) as the distance between the two TEM00 peaks. For now it just
    checks that the separation is bigger than 1V'.
    '''

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

    if (len(top_peaks) == 2):
        fsr_volt = np.abs(top_peaks[0][0] - top_peaks[1][0])
        return fsr_volt
    else:
        return None

def plot_voltage_vs_time(timestamps, volt_laser, volt_piezo, piezo_fitted, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, volt_laser, label='Volt Laser', color='blue')
    plt.plot(timestamps, volt_piezo, label='Volt Piezo', color='red')
    plt.plot(timestamps, piezo_fitted, label='Volt Piezo fitline', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage data vs Time')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()  # Uncomment this if you want to display the plot
    plt.close()  # Close the figure to avoid displaying it in-line

def plot_piezo_laser(piezo_fitted, volt_laser, xpeaks, ypeaks, file_name, width):
    plt.figure(figsize=(12, 6))
    plt.plot(piezo_fitted, volt_laser, label='Laser Intensity vs. Piezo volt', color='green', marker='.', linestyle=None)
    plt.hlines(ypeaks/2, xpeaks-width/2, xpeaks+width/2)
    plt.scatter(xpeaks, ypeaks, marker='x', label='Peak Values')
    plt.xlabel('Voltage Piezo (V)')
    plt.ylabel('Laser Intensity (V)')
    plt.title('Piezo Voltage vs Laser Voltage')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    #plt.show()
    plt.close()  # Close the figure to avoid displaying it

def plot_calibrated_laser(xvalues_freq, volt_laser, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(xvalues_freq, volt_laser, label='Laser Intensity vs. freq', color='green')
    plt.xlabel('relative freq values(Hz)( offset was set at 780nm ~ 384 THz )')
    plt.ylabel('Laser Intensity (V)')
    plt.title(' Laser Intensity (calibrated)')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    #plt.show()
    plt.close()  # Close the figure to avoid displaying it
