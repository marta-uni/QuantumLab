import matplotlib.pyplot as plt
import functions as fn


def plot_voltage_vs_time(timestamps, volt_laser, volt_piezo, piezo_fitted, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, volt_laser, label='Volt Laser', color='blue')
    plt.plot(timestamps, volt_piezo, label='Volt Piezo', color='red')
    plt.plot(timestamps, piezo_fitted,
             label='Volt Piezo fitline', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage data vs Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()  # Uncomment this if you want to display the plot
    plt.close()  # Close the figure to avoid displaying it in-line


def plot_piezo_laser(piezo_fitted, volt_laser, xpeaks, ypeaks, file_name, width=None):
    plt.figure(figsize=(12, 6))
    plt.plot(piezo_fitted, volt_laser, label='Laser Intensity vs. Piezo volt',
             color='green', marker='.', linestyle=None)
    if width is not None:
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
    plt.show()
    plt.close()  # Close the figure to avoid displaying it


def plot_calibrated_laser(xvalues_freq, volt_laser, file_name, extra_title=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(xvalues_freq, volt_laser, s=5,
             label='Laser Intensity vs. freq', color='green')
    plt.xlabel('Relative frequency values (GHz)')
    plt.ylabel('Laser Intensity (V)')
    plt.title(' Laser Intensity (calibrated)' + extra_title)
    plt.legend()
    plt.grid()
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(9,9), useOffset=False)
    plt.tight_layout()
    # plt.savefig(file_name)
    plt.show()
    plt.close()  # Close the figure to avoid displaying it


def plot_generic(x, y, x_label, y_label, file_name):
    coeffs_1, coeffs_2, x_fit, lin_fit, quad_fit = fn.lin_quad_fits(x, y)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data', color='green')
    plt.plot(x_fit, lin_fit, label='Linear Fit', color='blue',
             linestyle='--')  # Plot the quadratic fit
    plt.plot(x_fit, quad_fit, label='Quadratic Fit', color='red',
             linestyle='--')  # Plot the quadratic fit
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + ' vs ' + y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    #plt.savefig(file_name)
    plt.show()
    # plt.close()  # Close the figure to avoid displaying it


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
    plt.scatter(time[peaks_index], piezo[peaks_index],
                label='peaks', color='green')
    plt.plot(time, piezo, label='piezo data', color='blue',
             linestyle='--')  # Plot the quadratic fit
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
