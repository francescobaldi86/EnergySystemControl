import pandas as pd
import numpy as np

def read_timeseries_data_to_numpy(path, column_name: str|None = None):
    """
    This function reads data from a .csv file into a Numpy array
    The reason for this function is speed: dealing with numpy arrays is faster than pandas dataframes
    """
    data_df = pd.read_csv(path, sep = ';', decimal = '.', header = 0, index_col = 0, parse_dates = True)
    numpy_index = data_df.index.to_series().diff().total_seconds() / 3600  # Hours are used as standard time step unit
    numpy_data = data_df[column_name].values if column_name else data_df.values
    return np.array([numpy_index, numpy_data])

def C2K(T):
    return T + 273.15

def K2C(T):
    return T - 273.15

class NodeImbalanceError(Exception):
    pass

class MaxStorageError(Exception):
    pass