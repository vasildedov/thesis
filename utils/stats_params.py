def get_params(dataset, freq, model_type):
    asfreq = None

    # Consistent parameters across datasets
    if freq in ['Yearly', 'yearly']:
        order, seasonal_order, asfreq = (2, 1, 1), (0, 0, 0, 0), 'YE'
    elif freq in ['Quarterly', 'quarterly']:
        order, seasonal_order, asfreq = (2, 1, 1), (0, 1, 1, 4), 'QE'
    elif freq in ['Monthly', 'monthly']:
        order, seasonal_order, asfreq = (2, 1, 1), (0, 1, 1, 12), 'ME'
    elif freq in ['Weekly']:
        order, seasonal_order, asfreq = (1, 1, 1), (0, 1, 1, 52), None
    elif freq in ['Daily']:
        order, seasonal_order, asfreq = (1, 1, 1), (0, 1, 1, 7), None
    elif freq in ['Hourly']:
        order, seasonal_order, asfreq = (1, 1, 1), (0, 1, 1, 24), None
    elif freq in ['Other']:
        order, seasonal_order, asfreq = (1, 1, 1), (0, 0, 0, 0), None
    else:
        raise ValueError("Unsupported frequency. Choose a valid frequency.")

    # Adjust parameters for ARIMA to disable seasonal components
    if model_type == 'ARIMA':
        seasonal_order = (0, 0, 0, 0)  # Explicitly disable seasonal components for ARIMA

    # Return parameters based on the dataset and frequency
    return order, seasonal_order, asfreq