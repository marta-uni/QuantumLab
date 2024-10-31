import numpy as np
from pandas.core.common import is_full_slice
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def crop_to_min_max(time, laser_voltage, piezo_voltage):

    if len(piezo_voltage) == 0:
        return None  # Return None if piezo_voltage is empty

    # Find indices of minimum and maximum values
    min_index = np.argmin(piezo_voltage)
    max_index = np.argmax(piezo_voltage)

    # Ensure min_index is before max_index
    start_index = min(min_index, max_index)
    end_index = max(min_index, max_index)

    # Extract the data between min_index and max_index
    time_window = time[start_index:end_index + 1]
    laser_window = laser_voltage[start_index:end_index + 1]
    piezo_window = piezo_voltage[start_index:end_index + 1]

    return time_window, laser_window, piezo_window


def fit_piezo_line(time, piezo_voltage):

    if len(time) == 0 or len(piezo_voltage) == 0 or len(time) != len(piezo_voltage):
        return None  # Return None if the input arrays are empty or of different lengths

    # Fit a line (degree 1 polynomial) to the piezo voltage data
    slope, intercept = np.polyfit(time, piezo_voltage, 1)
    piezo_fit = slope * time + intercept

    return piezo_fit

def peaks(piezo_voltage, laser_voltage):
    peaks_indices, _ = find_peaks(laser_voltage, height= 0.3, distance=10)
    peaks = laser_voltage[peaks_indices]
    peaks_xvalues = piezo_voltage[peaks_indices]

    return peaks_xvalues, peaks

def plot_voltage_vs_time(timestamps, volt_laser, volt_piezo, piezo_fitted, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, volt_laser, label='Volt Laser', color='blue')
    plt.plot(timestamps, volt_piezo, label='Volt Piezo', color='red')
    plt.plot(timestamps, piezo_fitted, label='Volt Piezo fitline', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage data vs Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()  # Uncomment this if you want to display the plot
    plt.close()  # Close the figure to avoid displaying it in-line

def plot_piezo_laser(piezo_fitted, volt_laser, xpeaks, ypeaks, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(piezo_fitted, volt_laser, label='Laser Intensity vs. Piezo volt', color='green')
    plt.scatter(xpeaks, ypeaks, marker='x', label='Peak Values')
    plt.xlabel('Voltage Piezo (V)')
    plt.ylabel('Laser Intensity (V)')
    plt.title('Piezo Voltage vs Laser Voltage')
    plt.legend()
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
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    #plt.show()
    plt.close()  # Close the figure to avoid displaying it
