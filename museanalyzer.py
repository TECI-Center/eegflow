
from collections import defaultdict
from datetime import datetime

import numpy as np
from eeg_functions import compute_focus_index, compute_engagement_index, compute_FAA_index, compute_TLX
import importlib.util, pathlib
_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("eegflow_config", _cfg_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
config = _mod.config

class musemetrics(object):
    def __init__(self, annotations, museobject, pid, config=config):
        self.muse = museobject
        self.data = self.muse.data
        self.config = config
        self.annotations = annotations
        self.pid = pid
        self.metric_time_series = {}
        self.aggreate_metrics = {}
    
    def calculate_time_series(self):
        fs = self.config['fs']
        bb = self.config['beta_band']
        ab = self.config['alpha_band']
        tb = self.config['theta_band']
        db = self.config['delta_band']
        t1, fi = compute_focus_index(self.data, fs, bb, ab, start_time=self.data['ts'])
        t1, ei = compute_engagement_index(self.data, fs, bb, ab, tb, start_time=self.data['ts'])
        t2, faa = compute_FAA_index(self.data, fs, ab, start_time=self.data['ts'])
        t3, tlx = compute_TLX(self.data, fs, ab, tb, start_time=self.data['ts'])
        self.metric_time_series['focus_index'] = (t1, fi)
        self.metric_time_series['engagement_index'] = (t1, ei)
        self.metric_time_series['FAA_index'] = (t2, faa)
        self.metric_time_series['TLX'] = (t3, tlx)
        # TODO Check with Cameron
        self.metric_time_series['time'] = t1 # all time series should be the same length and aligned, so we can just use one of them as the reference time??

    def compute_metrics(self):
        for phase in self.annotations:
            start_time = self.annotations[phase][0]
            end_time = self.annotations[phase][1]
            self.aggreate_metrics[phase] = {
                'focus_index': np.mean(self.get_data_in_phase(self.metric_time_series['focus_index'][1], self.metric_time_series['time'], start_time, end_time)[0]),
                'engagement_index': np.mean(self.get_data_in_phase(self.metric_time_series['engagement_index'][1], self.metric_time_series['time'], start_time, end_time)[0]),
                'FAA_index': np.mean(self.get_data_in_phase(self.metric_time_series['FAA_index'][1], self.metric_time_series['time'], start_time, end_time)[0]),
                'TLX': np.mean(self.get_data_in_phase(self.metric_time_series['TLX'][1], self.metric_time_series['time'], start_time, end_time)[0])}
        return self.aggreate_metrics

    def get_data_in_phase(self, series, time, start, stop):
        """
        Filters the input series to include only data points where the corresponding time is between start and stop.
        series: A list or numpy array of data points.
        time: A list or numpy array of datetimes corresponding to each data point in series.
        start: The start time of the phase. epoch
        stop: The end time of the phase. epoch
        """ 
        if len(series) == 0:
            return series, time  # empty input, return as is

        # Boolean mask for times within range
        time_epoch = np.array(list(map(lambda t: datetime.timestamp(t), time)))  # convert datetime to epoch seconds
        mask = (time_epoch >= start) & (time_epoch <= stop)

        # Return subset
        return series[mask], time[mask]
