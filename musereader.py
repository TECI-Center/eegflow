"""
Class that reads music files and extracts EEG data
"""

from datetime import datetime

import numpy as np
from collections import defaultdict
import os
from eeg_functions import filter_raw_data
import pandas as pd
import importlib.util, pathlib
_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("eegflow_config", _cfg_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
config = _mod.config

class musereader(object):
    def __init__(self, filename, pid='test', conifg=config):
        self.filename = filename
        self.data = defaultdict(list)
        self.pid = None
        self.config = config
        self.read_eeg(filename)

    def read_eeg(self, fp):
        df = pd.read_csv(fp, 
                    sep=" ",          # split on any whitespace
                    header=None,         # no header in file
                    engine="python", 
                    names=range(7))
        df.columns = ["ts", "path", "dtype", "val1", "val2", "val3", "val4"]

        # Filter for only /eeg/ rows
        eeg_df = df[df["path"] == "/eeg/"].copy()
        
        # Rename channels
        eeg_df.rename(columns={
            "val1": "ch1",
            "val2": "ch2",
            "val3": "ch3",
            "val4": "ch4"
        }, inplace=True)
        
        # Convert timestamp to float epoch seconds
        eeg_df["ts"] = eeg_df["ts"].astype(float)
        
        # (Optional) Convert to datetime
        eeg_df['datetime'] = (
        pd.to_datetime(eeg_df['ts'], unit='s', utc=True)
        .dt.tz_convert('America/Los_Angeles')
        .dt.strftime('%Y-%m-%d %H:%M:%S')
    )
        # eeg_df["datetime"] = pd.to_datetime(eeg_df["ts"], unit="s").dt.tz_localize('America/Los_Angeles')
        
        # Reorder columns for convenience
        eeg_data_cleaned = pd.DataFrame(eeg_df[["ts", "datetime", "ch1", "ch2", "ch3", "ch4"]])
        eeg_data_cleaned = pd.DataFrame(filter_raw_data(eeg_data_cleaned, config['fs'], config['lowcut'], config['highcut']))
        self.data = eeg_data_cleaned
        # return eeg_data_cleaned

    # Function to read data from a file
    def read(self, hx=False, eeg_path=None):
        with open(eeg_path, 'r') as file:
            data = file.read()
        return self._process_data(data, hx)

    def _process_data(self, data, hx):
        output_list = []

        for line in data.splitlines():
            parts = line.split()
            
            # Decode the first part (timestamp in hexadecimal) to datetime
            
            datetime_ts = self._decode_hex_timestamp(parts[0], hx)
            # Create a dictionary of the results
            entry = {
                "timestamp_hex": parts[0],
                "timestamp_decoded": datetime_ts,
                "type": parts[1],
                "flag": parts[2],
                "values": list(map(lambda x: float(x) if x!='nan' else 0, [i for i in parts[3:]]))
            }
            output_list.append(entry)

        self.data['eeg'] = np.array([entry['values'] for entry in output_list if entry['type'] == '/eeg/'])
        self.data['time'] = np.array([entry['timestamp_decoded'] for entry in output_list if entry['type'] == '/eeg/'])       
        return output_list


    def _decode_hex_timestamp(self, hex_timestamp, hx):
        """
        This function takes a hexadecimal timestamp in the format 'ea8463dc.a97be984',
        splits it by the decimal point, converts both parts to integers, converts to datetime 
        and returns it
        
        Args:
        hex_timestamp (str): The hexadecimal timestamp string.
        
        Returns:
        datetime object: 
        """
        if hx: 
            # First part is the timestamp (hex), convert to float (hex part before '/')
            timestamp = float(int(hex_timestamp.split('.')[0], 16)) + float(int(hex_timestamp.split('.')[1], 16)) / (16**8)
        else:
            timestamp = float(hex_timestamp)
        datetime_ts = datetime.fromtimestamp(timestamp)
        return datetime_ts