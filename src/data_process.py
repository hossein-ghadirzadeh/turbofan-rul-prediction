import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

INDEX_NAMES = ['unit_nr', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES = ['s_{}'.format(i) for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

SELECTED_FEATURES = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
                     's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

def load_data(train_path: str, test_path: str, rul_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw CMAPSS data from txt files.
    """
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COL_NAMES)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COL_NAMES)
    rul_truth = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    return train_df, test_df, rul_truth

def process_targets(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Calculates RUL (Remaining Useful Life).
    We clip the RUL at 'max_rul' (e.g., 125) because in early stages, 
    degradation is not visible, and we don't want the model to learn noise.
    """
    # 1. Calculate max cycle for each unit
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max']
    
    # 2. Merge back
    df = df.merge(max_cycle, on=['unit_nr'], how='left')
    
    # 3. Calculate RUL
    df['RUL'] = df['max'] - df['time_cycles']
    
    # 4. Clip RUL (Piecewise Linear RUL)
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    
    df.drop('max', axis=1, inplace=True)
    return df

def scale_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Normalizes sensor data using MinMax Scaler based on TRAIN data only to avoid leakage.
    """
    scaler = MinMaxScaler()
    
    train_df[SELECTED_FEATURES] = scaler.fit_transform(train_df[SELECTED_FEATURES])
    test_df[SELECTED_FEATURES] = scaler.transform(test_df[SELECTED_FEATURES])
    
    return train_df, test_df, scaler

def gen_sequence(id_df: pd.DataFrame, seq_length: int, seq_cols: List[str]) -> np.ndarray:
    """
    Generates time-series windows for LSTM.
    Shape: (Samples, Time_Steps, Features)
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    
    # Iterate and create windows
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df: pd.DataFrame, seq_length: int, label: List[str]) -> np.ndarray:
    """
    Generates corresponding labels (RUL) for the sequences.
    """
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    
    return data_matrix[seq_length:num_elements, :]

def make_sequences(df: pd.DataFrame, seq_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to transform DataFrame into 3D Numpy Arrays for LSTM.
    """
    seq_gen = (list(gen_sequence(df[df['unit_nr'] == id], seq_length, SELECTED_FEATURES)) 
               for id in df['unit_nr'].unique())
    
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    label_gen = (list(gen_labels(df[df['unit_nr'] == id], seq_length, ['RUL'])) 
                 for id in df['unit_nr'].unique())
    
    label_array = np.concatenate(list(label_gen)).astype(np.float32)
    
    return seq_array, label_array