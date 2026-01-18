import os
import hdf5plugin
import pandas as pd
import h5py

from pathlib import Path
from typing import Dict, Tuple, Union

import math
import cv2
import imageio
from numba import jit
import numpy as np
import weakref
import re

from torch.utils.data import Dataset
from data.utils.dsec_utils import VoxelGrid, flow_16bit_to_float
from data.utils.pad_events import *


def collate_fn(batch):
    batched_data = {'events': [], 'timestamp': [], 'voxel': [], 'file_index': []}
    max_events = 0
    all_have_gt_flow = True

    for sample in batch:
        # Find the longest event sample
        if len(sample['events']) > max_events:
            max_events = len(sample['events'])
        # Check if GT flow is available
        if 'forward_flow' not in sample:
            all_have_gt_flow = False

    if all_have_gt_flow:
        batched_data['forward_flow'] = []
        batched_data['flow_mask'] = []
        # batched_data['image'] = []

    for sample in batch:
        # Pad raw events
        padded_events = pad_events(sample['events'], max_events)

        # Collect all the data samples in this batch
        batched_data['events'].append(padded_events)
        batched_data['timestamp'].append(sample['timestamp'])
        batched_data['voxel'].append(sample['voxel'])
        batched_data['file_index'].append(sample['file_index'])
        if all_have_gt_flow:
            batched_data['forward_flow'].append(sample['forward_flow'])
            batched_data['flow_mask'].append(sample['flow_mask'])
            # batched_data['image'].append(sample['image'])

    # Stack all the samples into a big tensor
    batched_data['events'] = torch.stack(batched_data['events'], dim=0)
    batched_data['timestamp'] = torch.stack(batched_data['timestamp'], dim=0)
    batched_data['voxel'] = torch.stack(batched_data['voxel'], dim=0)
    batched_data['file_index'] = torch.stack(batched_data['file_index'], dim=0)

    if all_have_gt_flow:
        batched_data['forward_flow'] = torch.stack(batched_data['forward_flow'], dim=0)
        batched_data['flow_mask'] = torch.stack(batched_data['flow_mask'], dim=0)
        # batched_data['image'] = torch.stack(batched_data['image'], dim=0)

    return batched_data


def filter_strings(strings, patterns):
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    filtered_list = [s for s in strings if not any(pattern.match(s) for pattern in compiled_patterns)]
    return filtered_list


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us / 1000)
        window_end_ms = math.ceil(ts_end_us / 1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class Sequence(Dataset):
    def __init__(self, seq_path: Path, phase: str = 'train', num_bins: int = 15, transforms=None,
                 sub_factor=1, timestamp_path: str = None, polarity_aware_batching=False,
                 norm_type=None, quantile=0, seq_length=5):
        assert num_bins >= 1
        assert seq_path.is_dir()
        '''
        Directory Structure:

        Dataset
        └── test
            ├── interlaken_00_b
            │        ├── events_left
            │        │       ├── events.h5
            │        │       └── rectify_map.h5
            │        ├── image_timestamps.txt
            │        └── test_forward_flow_timestamps.csv

        '''
        self.name = seq_path.name
        self.phase = phase
        self.seq_length = seq_length

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.sub_factor = sub_factor
        self.polarity_aware_batching = polarity_aware_batching

        # Set event representation
        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width),
                                    norm_type=norm_type, quantile=quantile)

        # Left events only
        ev_dir_location = seq_path / 'events/left'
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        # Event slicer
        event_h5f = h5py.File(str(ev_data_file), 'r')
        self.event_h5f = event_h5f
        self.event_slicer = EventSlicer(event_h5f)

        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]

        # Save delta timestamp in ms
        self.delta_t_us = 1e5  # 0.1s

        # Load and compute timestamps and indices
        if timestamp_path is None:
            raise ValueError
        df = pd.read_csv(timestamp_path)
        flow_start_time = df['from_timestamp_us']
        flow_end_time = df['to_timestamp_us']
        self.timestamps_flow = np.stack((flow_start_time.to_numpy(), flow_end_time.to_numpy()), axis=1)
        self.indices = df['file_index']
        self.paths_to_forward_flow = None
        self.paths_to_images = None

        # Check the number of samples
        assert len(self.timestamps_flow) == len(self.indices)
        if self.paths_to_forward_flow is not None:
            assert len(self.timestamps_flow) == len(self.paths_to_forward_flow)
        if self.paths_to_images is not None:
            assert len(self.timestamps_flow) == len(self.paths_to_images)

        self._finalizer = weakref.finalize(self, self.close_callback, self.event_h5f)

    def events_to_voxel_grid(self, p, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).astype('float32')
        t = (t / t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32') / 256

    @staticmethod
    def load_flow(flow_file: Path):
        assert flow_file.exists()
        assert flow_file.suffix == '.png'
        flow_16bit = imageio.imread(str(flow_file), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def rectify_events(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_data_sample(self, index):
        # Get the start and end time of this sample
        t_start = self.timestamps_flow[index, 0]
        t_end = self.timestamps_flow[index, 1]
        file_index = self.indices[index]

        # Create the output sample (a dictionary)
        output = {
            'name': f'{self.name}_{str(int(file_index)).zfill(6)}',
            'timestamp': torch.tensor(self.timestamps_flow[index]),
            'file_index': torch.tensor(file_index, dtype=torch.int64),
        }

        # Get raw events
        event_data = self.event_slicer.get_events(t_start, t_end)
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        # Normalize event timestamps
        t = (t - t_start) / (t_end - t_start)

        # Rectify raw events
        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        # Remove the ones that go outsides the camera plane
        mask = ((0 <= y_rect) & (y_rect < self.height) & (0 <= x_rect) & (x_rect < self.width))
        x_rect = x_rect[mask]
        y_rect = y_rect[mask]
        t = t[mask]
        p = p[mask]
        events = np.column_stack((x_rect, y_rect, t, p))  # Normalized time is used
        events = events.astype('float32')

        # Convert the raw event packet into the voxel grid
        if self.voxel_grid is None:
            raise NotImplementedError
        else:
            event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
            output['voxel'] = event_representation

        # Return the raw event packet of this sample
        if self.polarity_aware_batching:
            output['pos_events'] = torch.from_numpy(events[events[:, 3] == 1])
            output['neg_events'] = torch.from_numpy(events[events[:, 3] == 0])
        else:
            output['events'] = torch.from_numpy(events)

        return output

    def __getitem__(self, idx):
        sample = self.get_data_sample(idx)
        return sample

    def __len__(self):
        return len(self.timestamps_flow)
