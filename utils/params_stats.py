def get_params(freq, model_type):
    # Define default parameters for each frequency
    freq_params = {
        'Yearly': ((2, 1, 1), (0, 0, 0, 0), 'YE'),
        'Quarterly': ((2, 1, 1), (0, 1, 1, 4), 'QE'),
        'Monthly': ((2, 1, 1), (0, 1, 1, 12), 'ME'),
        'Weekly': ((1, 1, 1), (0, 1, 1, 52), None),
        'Daily': ((1, 1, 1), (0, 1, 1, 7), None),
        'Hourly': ((1, 1, 1), (0, 1, 1, 24), None),
        'Other': ((1, 1, 1), (0, 0, 0, 0), None)
    }

    # Handle case insensitivity for frequency
    freq = freq.capitalize()
    if freq not in list(freq_params.keys()):
        raise ValueError("Unsupported frequency. Choose a valid frequency.")

    order, seasonal_order, asfreq = freq_params[freq]

    # Disable seasonal components for ARIMA models
    if model_type.upper() == 'ARIMA':
        seasonal_order = (0, 0, 0, 0)

    return order, seasonal_order, asfreq
