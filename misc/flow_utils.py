import numpy as np
import torch
from torch import nn


def interp_dense_flow_from_tiles(flow_tiles: torch.Tensor,
                                 output_height: int,
                                 output_width: int) -> torch.Tensor:
    """
    Bi-linear interpolate to generate high-res flow map from the low one.
    :param flow_tiles: (B, 2, H_input, W_input)
    :param output_height: H_output
    :param output_width: H_input
    :return: high-res flow map
    """

    # Batch size
    b = flow_tiles.shape[0]

    # Flow height and width (number of tiles)
    input_height = flow_tiles.shape[2]
    input_width = flow_tiles.shape[3]

    # Size of tiles
    tile_width = output_width / input_width
    tile_height = output_height / input_height


    # Pad the input flow tiles
    pad = nn.ReplicationPad2d(1)
    flow_tiles_padded = pad(flow_tiles)

    # Pixel coordinate of the output flow map
    x_range = np.arange(tile_width, output_width + tile_width)
    y_range = np.arange(tile_height, output_height + tile_height)
    pixels = np.array([[x, y] for y in y_range for x in x_range])
    pixels = torch.from_numpy(pixels).float()
    pixels = pixels.to(flow_tiles.device)
    # print(f"pixels: \n{pixels}")
    x = pixels[:, 0]
    x = x.repeat(b, 1)
    y = pixels[:, 1]
    y = y.repeat(b, 1)

    # Perform bi-linear interpolation to query optical flow for each pixel
    # Calculate the pixel locations in the coordinate of the input flow (tiles)
    tile_x = (x - tile_width / 2) / tile_width
    tile_y = (y - tile_height / 2) / tile_height

    # Get the locations of the top left involved tile
    tile_x0 = torch.floor(tile_x).int()
    tile_y0 = torch.floor(tile_y).int()

    # Compute dx and dy
    tile_dx = tile_x - tile_x0
    tile_dy = tile_y - tile_y0

    # The indices of the four involved tiles
    padded_input_width = input_width + 2
    F1_indices = tile_y0.long() * padded_input_width + tile_x0.long()
    F2_indices = (tile_y0 + 1).long() * padded_input_width + tile_x0.long()
    F3_indices = tile_y0.long() * padded_input_width + (tile_x0 + 1).long()
    F4_indices = (tile_y0 + 1).long() * padded_input_width + (tile_x0 + 1).long()

    # Get the flow values at the involved pixels
    flow_tiles_flat = flow_tiles_padded.reshape((b, 2, -1))
    Fx1 = torch.gather(flow_tiles_flat[:, 0], 1, F1_indices)
    Fx2 = torch.gather(flow_tiles_flat[:, 0], 1, F2_indices)
    Fx3 = torch.gather(flow_tiles_flat[:, 0], 1, F3_indices)
    Fx4 = torch.gather(flow_tiles_flat[:, 0], 1, F4_indices)
    Fy1 = torch.gather(flow_tiles_flat[:, 1], 1, F1_indices)
    Fy2 = torch.gather(flow_tiles_flat[:, 1], 1, F2_indices)
    Fy3 = torch.gather(flow_tiles_flat[:, 1], 1, F3_indices)
    Fy4 = torch.gather(flow_tiles_flat[:, 1], 1, F4_indices)

    # Bi-linear interpolation
    tile_w1 = (1.0 - tile_dx) * (1.0 - tile_dy)
    tile_w2 = (1.0 - tile_dx) * tile_dy
    tile_w3 = tile_dx * (1.0 - tile_dy)
    tile_w4 = tile_dx * tile_dy
    flow_x = Fx1 * tile_w1 + Fx2 * tile_w2 + Fx3 * tile_w3 + Fx4 * tile_w4
    flow_y = Fy1 * tile_w1 + Fy2 * tile_w2 + Fy3 * tile_w3 + Fy4 * tile_w4
    output_flow = torch.cat([flow_x.reshape((b, 1, output_height, output_width)),
                             flow_y.reshape((b, 1, output_height, output_width))], dim=1)
    return output_flow


def calculate_flow_error(
    flow_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_mask: torch.Tensor = None,
    event_mask: torch.Tensor = None,
    time_scale: torch.Tensor = None,
) -> dict:
    """Calculate flow error.
    Args:
        flow_gt (torch.Tensor) ... [B x 2 x H x W]
        flow_pred (torch.Tensor) ... [B x 2 x H x W]
        flow_mask (torch.Tensor) ... [B x 1 x W x H]. Optional. If none, compute it.
        event_mask (torch.Tensor) ... [B x 1 x W x H]. Optional.
        time_scale (torch.Tensor) ... [B x 1]. Optional. This will be multiplied.
            If you want to get error in 0.05 ms, time_scale should be
            `0.05 / actual_time_period`.
    Returns:
        errors (dict) ... Key containers 'AE', 'EPE', '1/2/3PE'. all float.

    """
    if flow_mask is None:
        # Compute a flow mask
        # Only compute error over points that are valid in the GT (not inf or 0).
        flow_mask = torch.logical_and(
            torch.logical_and(~torch.isinf(flow_gt[:, [0], ...]), ~torch.isinf(flow_gt[:, [1], ...])),
            torch.logical_and(torch.abs(flow_gt[:, [0], ...]) > 0, torch.abs(flow_gt[:, [1], ...]) > 0),
        )  # B, H, W
    if event_mask is None:
        total_mask = flow_mask
    else:
        if len(event_mask.shape) == 3:
            event_mask = event_mask[:, None]
        total_mask = torch.logical_and(event_mask, flow_mask)
    gt_masked = flow_gt * total_mask  # b, 2, H, W
    pred_masked = flow_pred * total_mask
    n_points = torch.sum(total_mask, dim=(1, 2, 3)) + 1e-5  # B, 1

    errors = {}
    # Average endpoint error.
    if time_scale is not None:
        time_scale = time_scale.reshape(len(gt_masked), 1, 1, 1)
        gt_masked = gt_masked * time_scale
        pred_masked = pred_masked * time_scale
    endpoint_error = torch.linalg.norm(gt_masked - pred_masked, dim=1)
    errors["EPE"] = torch.mean(torch.sum(endpoint_error, dim=(1, 2)) / n_points)
    errors["1PE"] = torch.mean(torch.sum(endpoint_error > 1, dim=(1, 2)) / n_points)
    errors["2PE"] = torch.mean(torch.sum(endpoint_error > 2, dim=(1, 2)) / n_points)
    errors["3PE"] = torch.mean(torch.sum(endpoint_error > 3, dim=(1, 2)) / n_points)

    # Angular error
    u, v = pred_masked[:, 0, ...], pred_masked[:, 1, ...]
    u_gt, v_gt = gt_masked[:, 0, ...], gt_masked[:, 1, ...]
    cosine_similarity = (1.0 + u * u_gt + v * v_gt) / (torch.sqrt(1 + u * u + v * v) * torch.sqrt(1 + u_gt * u_gt + v_gt * v_gt))
    cosine_similarity = torch.clamp(cosine_similarity, -1, 1)
    errors["AE"] = torch.mean(torch.sum(torch.acos(cosine_similarity), dim=(1, 2)) / n_points)
    errors["AE"] = errors["AE"] * (180.0 / torch.pi)
    return errors
