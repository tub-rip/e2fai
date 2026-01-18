import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.unet import UNet
from misc.flow_utils import interp_dense_flow_from_tiles


class IntensityFlowNet(pl.LightningModule):
    def __init__(self, sensor_height, sensor_width,
                 num_bins, bilinear=False, flow_tile_size=16):
        super(IntensityFlowNet, self).__init__()

        # Flow pooling
        self.flow_pooling = nn.AvgPool2d(flow_tile_size)

        # Shape of input/output
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        self.num_bins = num_bins

        # Network settings
        self.model = UNet(self.num_bins, 3, bilinear=bilinear)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def predict_step(self, batch, batch_idx):
        flow, intensity = self(batch)
        return flow.detach(), intensity.detach()

    def forward(self, batch):
        # Get the event voxel grid
        voxel = batch['voxel']  # [N, C, H, W]

        # Predict the flow and intensity through the network
        output = self.model(voxel)
        flow = output[:, :2, ...]
        intensity = output[:, 2:, ...]

        # Flow pooling
        flow = self.flow_pooling(flow)

        # Interpolate full-resolution flow from the pooling one
        flow = interp_dense_flow_from_tiles(flow, self.sensor_height, self.sensor_width)

        return flow, intensity

    def configure_optimizers(self):
        pass
