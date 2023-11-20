import numpy as np
import matplotlib.pyplot as plt

def autocovariance_plot(time_series, max_lag=None):
    """
    Visualizes the autocovariance of a time series.

    Parameters:
    - time_series (array-like): The input time series data.
    - max_lag (int, optional): Maximum lag for which to compute autocovariance. 
      If None, it defaults to half of the length of the time series.

    Returns:
    - None: Displays the autocovariance plot.
    """

    # Validate input
    if not isinstance(time_series, (list, np.ndarray)):
        raise ValueError("Input 'time_series' must be a list or numpy array.")

    if len(time_series) < 2:
        raise ValueError("Input 'time_series' must have at least two elements.")

    if max_lag is not None and not isinstance(max_lag, int):
        raise ValueError("'max_lag' must be an integer if provided.")

    # Set default max_lag if not provided
    if max_lag is None:
        max_lag = len(time_series) // 2

    # Compute autocovariance
    autocovariance = np.correlate(time_series - np.mean(time_series), time_series - np.mean(time_series), mode='full') / len(time_series)

    # Normalize autocovariance to get autocorrelation
    autocorrelation = autocovariance / autocovariance[max_lag]

    # Plot autocorrelation function
    plt.figure(figsize=(10, 6))
    lags = np.arange(-max_lag, max_lag + 1)
    plt.stem(lags, autocorrelation)
    plt.title('Autocovariance Plot')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()

# Example usage:
# Generate a sample time series
np.random.seed(42)
time_series = np.random.randn(100)

# Visualize autocovariance
autocovariance_plot(time_series,10)