def get_params(dataset, freq, model_type):
    max_length = None
    asfreq = None

    # Consistent parameters across datasets
    if freq in ['Yearly', 'yearly']:
        order, seasonal_order, max_length, asfreq = (2, 1, 1), (0, 0, 0, 0), None, 'YE'
    elif freq in ['Quarterly', 'quarterly']:
        order, seasonal_order, max_length, asfreq = (2, 1, 1), (0, 1, 1, 4), None, 'QE'
    elif freq in ['Monthly', 'monthly']:
        order, seasonal_order, max_length, asfreq = (2, 1, 1), (0, 1, 1, 12), 120, 'ME'
    elif freq in ['Weekly', 'weekly']:
        order, seasonal_order, max_length, asfreq = (1, 1, 1), (0, 0, 0, 52), None, None
    elif freq in ['Daily', 'daily']:
        order, seasonal_order, max_length, asfreq = (1, 1, 1), (0, 0, 0, 7), 200, None
    elif freq in ['Hourly', 'hourly']:
        order, seasonal_order, max_length, asfreq = (1, 1, 1), (0, 1, 1, 24), None, None
    elif freq in ['Other', 'other']:
        order, seasonal_order, max_length, asfreq = (1, 1, 1), (0, 0, 0, 0), None, None
    else:
        raise ValueError("Unsupported frequency. Choose a valid frequency.")

    # Adjust parameters for ARIMA to disable seasonal components
    if model_type == 'ARIMA':
        seasonal_order = (0, 0, 0, 0)  # Explicitly disable seasonal components for ARIMA

    # Return parameters based on the dataset and frequency
    return order, seasonal_order, max_length if dataset == 'm4' else asfreq