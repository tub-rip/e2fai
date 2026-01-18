import argparse
from pathlib import Path
import os
import sys
from datetime import datetime
import PIL.Image as Image

import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.intensity_flow_net import IntensityFlowNet
from data.loader.loader_dsec import Sequence, collate_fn
from misc.visualizer import visualize_optical_flow, visualize_intensity

torch.set_float32_matmul_precision('high')


def make_event_image(x, y, p, height, width):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image = make_overlay_image(image, x.int(), y.int(), p.int())
    return image


def make_overlay_image(image, x, y, p):
    image[y[p == 0], x[p == 0]] = (255, 0, 0)
    image[y[p == 1], x[p == 1]] = (0, 0, 255)
    return image


def save_flow_eval(file_path: str, flow: np.ndarray):
    height, width = flow.shape[1], flow.shape[2]
    flow_16bit = np.zeros((height, width, 3), dtype=np.uint16)
    flow_16bit[..., 0] = (flow[0] * 128 + 2 ** 15).astype(np.uint16)  # x-component
    flow_16bit[..., 1] = (flow[1] * 128 + 2 ** 15).astype(np.uint16)  # y-component
    imageio.imwrite(file_path, flow_16bit, format='PNG-FI')


def scale_optical_flow(flow, max_flow_magnitude):
    u, v = flow[0, :, :], flow[1, :, :]
    magnitude = torch.sqrt(u ** 2 + v ** 2)
    exceed_indices = magnitude > max_flow_magnitude

    u_scaled = u.clone()
    v_scaled = v.clone()

    u_scaled[exceed_indices] = (u[exceed_indices] / magnitude[exceed_indices]) * max_flow_magnitude
    v_scaled[exceed_indices] = (v[exceed_indices] / magnitude[exceed_indices]) * max_flow_magnitude
    scaled_flow = torch.stack([u_scaled, v_scaled], dim=0)

    return scaled_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckp_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation/dsec/data')
    parser.add_argument('--dataset_dir', type=str, default='/home/datasets/dsec')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='benchmark')
    args = parser.parse_args()

    if args.mode in ['benchmark', 'val_all']:
        dataset_dir = os.path.join(args.dataset_dir, 'test')
        timestamp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '../evaluation/dsec/test_timestamps/pd')
        test_seqs = [p[:-4] for p in os.listdir(timestamp_dir) if p.endswith('.csv')]
    elif args.mode == 'train_all':
        dataset_dir = os.path.join(args.dataset_dir, 'train')
        test_seqs = os.listdir(dataset_dir)
    else:
        raise ValueError

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Sensor size
    sensor_height = config['data']['height']
    sensor_width = config['data']['width']
    num_bins = config['data']['num_bins']

    # Model settings
    bilinear = config['model']['bilinear']

    # Device
    if torch.cuda.is_available() and args.gpu is not None:
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load model weights
    model = IntensityFlowNet.load_from_checkpoint(args.ckp_path,
                                                  sensor_height=sensor_height, sensor_width=sensor_width,
                                                  num_bins=num_bins, bilinear=bilinear, map_location=device)
    model.eval()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_out_dir = os.path.join(args.output_dir, f'{timestamp}_{args.name}')

    with torch.no_grad():
        for seq in test_seqs:
            print(f'Processing: {seq}')
            # Input dataset dir
            seq_dir = os.path.join(dataset_dir, seq)
            # Output results dir
            event_image_dir = os.path.join(run_out_dir, 'event_image', seq)
            os.makedirs(event_image_dir)
            flow_eval_dir = os.path.join(run_out_dir, 'flow_eval', seq)
            os.makedirs(flow_eval_dir)
            flow_rgb_dir = os.path.join(run_out_dir, 'flow_rgb', seq)
            os.makedirs(flow_rgb_dir)
            intensity_dir = os.path.join(run_out_dir, 'intensity', seq)
            os.makedirs(intensity_dir)
            intensity_numpy_dir = os.path.join(run_out_dir, 'intensity_numpy', seq)
            os.makedirs(intensity_numpy_dir)

            # Define the dataloader
            if args.mode == 'benchmark':
                timestamp_path = os.path.join(timestamp_dir, f'{seq}.csv')
                dataset = Sequence(Path(seq_dir), phase='test', num_bins=15,
                                   timestamp_path=timestamp_path)
            elif args.mode in ['val_all', 'train_all']:
                dataset = Sequence(Path(seq_dir), phase='train', num_bins=15)
            else:
                raise ValueError

            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

            # Loop through all samples
            for sample in tqdm(loader):
                sample['voxel'] = sample['voxel'].to(model.device)

                # Get the file index
                file_index = sample['file_index'].item()
                file_name = f'{str(file_index).zfill(6)}.png'

                # Visualize event image
                events = sample['events']
                raw_event_image = make_event_image(events[:, :, 0], events[:, :, 1], events[:, :, 3],
                                                   sensor_height, sensor_width)
                event_image = Image.fromarray(raw_event_image)
                event_image_path = os.path.join(event_image_dir, file_name)
                event_image.save(event_image_path)

                # Predict flow and intensity using the model
                flow, intensity = model.predict_step(sample, 0)
                flow = flow.squeeze().cpu().numpy()
                intensity = intensity[:, 0, ...]  # Just in case, the network outputs multi-ref frames
                intensity = intensity.squeeze().cpu()

                # Save flow for evaluation
                file_path = os.path.join(flow_eval_dir, file_name)
                save_flow_eval(file_path, flow)

                # Save RGB flow for visualization
                flow_rgb = visualize_optical_flow(flow)
                flow_rgb = Image.fromarray(flow_rgb)
                file_path = os.path.join(flow_rgb_dir, file_name)
                flow_rgb.save(file_path)

                # Save intensity images
                # Perform exp-scale operation to convert to linear-scale
                intensity = torch.exp(intensity)  # Convert to linear scale
                arr_name = f'{str(file_index).zfill(6)}.npy'
                file_path = os.path.join(intensity_numpy_dir, arr_name)
                np.save(file_path, intensity.numpy())
                intensity = visualize_intensity(intensity, robust=True, percentile=2.0)
                intensity = Image.fromarray(intensity)
                file_path = os.path.join(intensity_dir, file_name)
                intensity.save(file_path)

    print('Done!')


if __name__ == '__main__':
    main()
