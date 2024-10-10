from TSForecasting.utils.data_loader import convert_tsf_to_dataframe

loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")
