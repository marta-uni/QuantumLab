from matplotlib.collections import QuadMesh
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

    return time, piezo_fit


def peaks_fsr(piezo_voltage, laser_voltage):

    # Use find_peaks with specified height and distance
    peaks_indices, _ = find_peaks(laser_voltage, height=0.2, distance=400)

    # Convert peaks_indices to an integer NumPy array
    peaks_indices = np.array(peaks_indices, dtype=int)
    print(peaks_indices)
    # Extract peak values
    peaks = laser_voltage[peaks_indices]
    peaks_xvalues = piezo_voltage[peaks_indices]

    return peaks_xvalues, peaks, peaks_indices


def peaks_hfsr(piezo_voltage, laser_voltage):

    # Use find_peaks with specified height and distance
    peaks_indices, _ = find_peaks(laser_voltage, height=0.08, distance=400)

    # Convert peaks_indices to an integer NumPy array
    peaks_indices = np.array(peaks_indices, dtype=int)
    print(peaks_indices)
    # Extract peak values
    peaks = laser_voltage[peaks_indices]
    peaks_xvalues = piezo_voltage[peaks_indices]

    return peaks_xvalues, peaks, peaks_indices


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


def save_to_csv(timestamps, volt_laser, volt_piezo, piezo_fitted, output_file):
    """
    Save the cropped and fitted data to a CSV file with the specified columns.

    Parameters:
    - timestamps: Array of timestamps
    - volt_laser: Array of laser voltages
    - volt_piezo: Array of piezo voltages
    - piezo_fitted: Array of fitted piezo voltage values
    - output_file: Path to the output CSV file
    """

    # Create a DataFrame with the data
    data = {
        'timestamp': timestamps,
        'volt_laser': volt_laser,
        'volt_piezo': volt_piezo,
        'piezo_fitted': piezo_fitted
    }

    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def lin_quad_fits(x,y):
    #perform linear fit
    coeffs_1 = np.polyfit(x, y, 1)  # coeffs = [a, b] for y = ax + b
    linear_fit = np.poly1d(coeffs_1)  # Create a polynomial function from the coefficients
    # Perform quadratic fit (polynomial of degree 2)
    coeffs_2 = np.polyfit(x, y, 2)  # coeffs = [a, b, c] for y = ax^2 + bx + c
    quadratic_fit = np.poly1d(coeffs_2)  # Create a polynomial function from the coefficients

    # Generate x values for the fitted curve (same x range as the original data)
    x_fit = np.linspace(min(x), max(x), 100)  # Smooth line for plotting
    lin_fit = linear_fit(x_fit)  # Calculate the fitted line
    quad_fit = quadratic_fit(x_fit)  # Calculate corresponding y values for the fitted curve
    print(coeffs_1)
    print(coeffs_2)
    return coeffs_1, coeffs_2, x_fit, lin_fit, quad_fit

    
def plot_generic(x, y, x_label, y_label, file_name):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data', color='green', s = 5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + ' vs ' + y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_name + ".pdf")
    # plt.show()
    plt.close()  # Close the figure to avoid displaying it

def plot_fits(x, y, x_label, y_label, file_name):
    coeffs_1, coeffs_2, x_fit, lin_fit, quad_fit = lin_quad_fits(x, y)

    # Format the coefficients for display
    lin_coeff_label = f"Linear Fit: y = ({coeffs_1[0]:.2e})x + ({coeffs_1[1]:.2e})"
    quad_coeff_label = (
        f"Quadratic Fit: y = ({coeffs_2[0]:.2e})x² + ({coeffs_2[1]:.2e})x + ({coeffs_2[2]:.2e})"
    )

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data', color='green', marker = 'x', s=30)
    plt.plot(x_fit, lin_fit, label=lin_coeff_label, color='blue', linestyle='--')  # Plot the linear fit
    plt.plot(x_fit, quad_fit, label=quad_coeff_label, color='red', linestyle='--')  # Plot the quadratic fit
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + ' vs ' + y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_name + ".pdf")
    # plt.show()
    plt.close()  # Close the figure to avoid displaying it
    return coeffs_1, coeffs_2

'''
def plot_generic(x, y, x_label, y_label, file_name):
    coeffs_1, coeffs_2, x_fit, lin_fit, quad_fit = lin_quad_fits(x, y)
    y_evenly_spaced = np.linspace(min(y), max(y), len(y))  # Generate evenly spaced y-values
    x_normalized = inverse_quadratic(coeffs_2[0], coeffs_2[1], coeffs_2[2], y_evenly_spaced)  # Inverse transformation

    # Create the plot
    plt.figure(figsize=(12, 6))
    new_data = 0.854*quad_fit - 0.92498884
    plt.scatter(x, y, label='Data', color='green')
    plt.plot(x_fit, lin_fit, label='Linear Fit', color='blue', linestyle='--')  # Plot the linear fit
    plt.plot(x_fit, quad_fit, label='Quadratic Fit', color='red', linestyle='--')  # Plot the quadratic fit
    plt.plot(x_normalized, y_evenly_spaced, label = 'prova', color = 'green', linestyle = '--' )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + ' vs ' + y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()
    plt.close()  # Close the figure to avoid displaying it
'''

def plot_scatter_generic(time, piezo, peaks_index, x_label, y_label, file_name):
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(time[peaks_index], piezo[peaks_index], label='peaks', color='green')
    plt.plot(time, piezo, label='piezo data', color='blue', linestyle='--')  # Plot the quadratic fit
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + ' vs ' + y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()
    plt.close()  # Close the figure to avoid displaying it
