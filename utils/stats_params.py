def get_params(dataset, freq, model_type):
    max_length = None
    if dataset == 'm4':
        if freq == 'Yearly':
            order, seasonal_order, max_length = (2, 1, 1), (1, 1, 0, 12), None
        elif freq == 'Quarterly':
            order, seasonal_order, max_length = (3, 1, 1), (1, 1, 0, 4), None
        elif freq == 'Monthly':
            order, seasonal_order, max_length = (6, 1, 1), (1, 1, 0, 12), 120
        elif freq == 'Weekly':
            order, seasonal_order, max_length = (5, 1, 1), (1, 1, 0, 52), None
        elif freq == 'Daily':
            order, seasonal_order, max_length = (5, 1, 1), (1, 1, 0, 7), 200
        elif freq == 'Hourly':
            order, seasonal_order, max_length = (24, 1, 1), (0, 1, 1, 24), None
        else:
            raise ValueError("Unsupported frequency. Choose a valid M4 frequency.")
    elif dataset == 'm3':
        if freq == 'Yearly':
            order, seasonal_order, asfreq = (1, 1, 0), (0, 0, 0, 0), 'YE'  # Minimal seasonality, focus on trend
        elif freq == 'Quarterly':
            order, seasonal_order, asfreq = (2, 1, 2), (0, 1, 0, 4), 'QE'  # Quarterly seasonality
        elif freq == 'Monthly':
            order, seasonal_order, asfreq = (3, 1, 2), (1, 1, 1, 12), 'ME'  # Monthly seasonality
        elif freq == 'Other':
            order, seasonal_order, asfreq = (1, 1, 1), (0, 0, 0, 0), None  # Minimal, adapt based on data
        else:
            raise ValueError("Unsupported frequency. Choose a valid M3 frequency.")
    else:
        if freq == 'yearly':
            order, seasonal_order, asfreq = (1, 1, 0), (0, 0, 0, 0), 'YE'  # Minimal seasonality, focus on trend
        elif freq == 'quarterly':
            order, seasonal_order, asfreq = (2, 1, 2), (0, 1, 1, 4), 'QE'  # Quarterly seasonality
        elif freq == 'monthly':
            order, seasonal_order, asfreq = (2, 1, 2), (1, 1, 1, 12), 'ME'  # Slightly simpler
        else:
            raise ValueError("Unsupported frequency. Choose a valid tourism frequency.")

    if model_type == 'ARIMA':
        seasonal_order = (0, 0, 0, 0)  # Explicitly set to zero-seasonality

    return order, seasonal_order, max_length
