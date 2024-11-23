from scipy.signal import find_peaks, peak_widths
import pandas as pd
import functions as fn
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data = pd.read_csv('data5/scope_10.csv', skiprows=2, names=['timestamp', 'volt_laser', 'volt_piezo'],
                   dtype={'timestamp': float, 'volt_laser': float, 'volt_piezo': float})

# clean data
data = data.dropna()

# Extract the columns
timestamps = data['timestamp'].to_numpy()  # Convert to NumPy array
# Convert to NumPy array
volt_laser = data['volt_laser'].to_numpy()
# Convert to NumPy array
volt_piezo = data['volt_piezo'].to_numpy()

# crop data to one piezo cycle
[timestamps, volt_laser, volt_piezo] = fn.crop_to_min_max(
    timestamps, volt_laser, volt_piezo)

# fit the piezo data
piezo_fitted = fn.fit_piezo_line(timestamps, volt_piezo)
piezo_spacing = np.mean(np.diff(piezo_fitted))

# Find peaks and widths
peaks, _ = find_peaks(volt_laser, height=0.1, distance=400)
widths_half = peak_widths(volt_laser, peaks, rel_height=0.5)[0]

# Define the Lorentzian function


def lorentzian(piezo_fitted, A, x0, gamma, off):
    return A / (1 + ((piezo_fitted - x0) / gamma) ** 2) + off


# Loop through each peak and fit
fitted_curves = []
for peak, width in zip(peaks, widths_half):
    # Determine a fitting range around the peak, e.g., ±1.2 * width
    fit_range = int(width * 1.2)
    start = max(0, peak - fit_range)
    end = min(len(piezo_fitted), peak + fit_range)

    # Extract data around the peak
    piezo_fit_range = piezo_fitted[start:end]
    volt_laser_fit_range = volt_laser[start:end]
    width_scaled = width * piezo_spacing

    # Initial guess: A=height at peak, x0=peak position in piezo_fitted, gamma=half-width at half-maximum
    initial_guess = [volt_laser[peak], piezo_fitted[peak], width_scaled / 2, 0]

    # Define bounds for A, x0, and gamma
    bounds = (
        # Lower bounds for [A, x0, gamma]
        [0, piezo_fitted[peak] - width_scaled, 0, 0],
        # Upper bounds for [A, x0, gamma]
        [np.inf, piezo_fitted[peak] + width_scaled, width_scaled * 2, volt_laser[peak] / 15]
    )

    try:
        # Fit the Lorentzian to the data in the fitting range with bounds
        popt, pcov = curve_fit(lorentzian, piezo_fit_range, volt_laser_fit_range,
                               p0=initial_guess, bounds=bounds, maxfev=10000)
        fitted_curve = lorentzian(piezo_fit_range, *popt)

        # Calculate residuals (data - fitted model)
        residuals = volt_laser_fit_range - fitted_curve

        # Print residuals and the sum of squared residuals (SSR) for fit quality
        SSR = np.sum(residuals**2)
        print(
            f"\nPeak at piezo_fitted = {piezo_fitted[peak]:.4f} fitted with parameters:")
        print(f"A = {popt[0]:.4f} +/- {np.sqrt(pcov[0,0]):.4f} V")
        print(f"x0 = {popt[1]:.4f} +/- {np.sqrt(pcov[1,1]):.4f} V")
        print(f"gamma = {popt[2]:.4f} +/- {np.sqrt(pcov[2,2]):.4f} V")
        print(f"off = {popt[3]:.4f} +/- {np.sqrt(pcov[3,3]):.4f} V")
        print(f"Sum of squared residuals (SSR): {SSR:.2f}")

        # Optionally, print the parameter covariance matrix
        print(f"Covariance matrix:\n{pcov}")

        # Store the fitted curve
        fitted_curves.append((piezo_fit_range, fitted_curve))

    except RuntimeError as e:
        print(
            f"Failed to fit peak at piezo_fitted = {piezo_fitted[peak]:.2f} due to RuntimeError: {e}")
    except Exception as e:
        print(
            f"An unexpected error occurred while fitting peak at piezo_fitted = {piezo_fitted[peak]:.2f}: {e}")

# Plot original data with fitted Lorentzians
plt.plot(piezo_fitted, volt_laser, marker='.', label='Original Signal')
for i, (piezo_fit_range, fitted_curve) in enumerate(fitted_curves):
    plt.plot(piezo_fit_range, fitted_curve, '--',
             label=f'Fitted Lorentzian {i+1}')
# Plot peaks with piezo_fitted coordinates
plt.plot(piezo_fitted[peaks], volt_laser[peaks], "x", label="Peaks")
plt.legend()
plt.grid()
plt.show()
