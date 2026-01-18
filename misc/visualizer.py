import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch


def visualize_color_wheel(image_height: int) -> np.ndarray:
    """ Visualize the color wheel.
    Args:
        image_height (int): Height of the color wheel image.
    Returns:
        color wheel image in RGB (PIL image).
    """
    # Color wheel
    hsv = np.zeros([image_height, image_height, 3], dtype=np.uint8)
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, image_height), np.linspace(-1, 1, image_height)
    )
    mag = np.linalg.norm(np.stack((xx, yy), axis=2), axis=2)
    ang = (np.arctan2(yy, xx) + np.pi) * 180 / np.pi / 2.0
    hsv[:, :, 0] = ang.astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (255 * mag / mag.max()).astype(np.uint8)
    color_wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # color_wheel_image = Image.fromarray(color_wheel)
    return color_wheel


def color_optical_flow(flow_x: np.ndarray, flow_y: np.ndarray, max_magnitude=None, ord=1.0):
    """ Color optical flow.
    Args:
        flow_x (np.ndarray)             ... [H x W], width direction.
        flow_y (np.ndarray)             ... [H x W], height direction.
        max_magnitude (float, optional) ... Max magnitude used for the colorization. Defaults to None.
        ord (float)                     ... 1: our usual, 0.5: DSEC colorizing.

    Returns:
        flow_rgb_image (np.ndarray)     ... Ready for showing or saving
        max_magnitude (float)           ... Max magnitude of the flow.
    """
    flows = np.stack((flow_x, flow_y), axis=2)
    flows[np.isinf(flows)] = 0
    flows[np.isnan(flows)] = 0
    mag = np.linalg.norm(flows, axis=2) ** ord
    ang = (np.arctan2(flow_y, flow_x) + np.pi) * 180.0 / np.pi / 2.0
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)

    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    if max_magnitude is None:
        max_magnitude = mag.max()
    hsv[:, :, 2] = (255 * mag / max_magnitude).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow_rgb, max_magnitude


def visualize_optical_flow(flow, ord=1.0) -> np.ndarray:
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().detach().numpy()
    if len(flow.shape) == 4:
        flow = np.squeeze(flow)
    """ Visualize optical flow in RGB images."""
    flow_rgb, _ = color_optical_flow(flow[0, :, :], flow[1, :, :], ord=ord)
    # flow_rgb_image = Image.fromarray(flow_rgb)
    return flow_rgb


def standard_normalize(intensity: np.ndarray) -> np.ndarray:
    I_min = intensity.min()
    I_max = intensity.max()
    image = (intensity - I_min) / (I_max - I_min)
    image = (255 * image).astype(np.uint8)
    return image


def robust_normalize(intensity: np.ndarray, percentile: float) -> np.ndarray:
    """ Normalize the intensity array robustly."""
    # Find the robust min & max values
    intensity_flat = intensity.flatten()
    intensity_sorted = np.sort(intensity_flat)
    robust_min_idx = int((0.5 * percentile / 100.0) * intensity_sorted.size)
    robust_max_idx = int((1 - 0.5 * percentile / 100.0) * intensity_sorted.size) - 1  # -1 for safety when percentile = 0
    robust_min_val = intensity_sorted[robust_min_idx]
    robust_max_val = intensity_sorted[robust_max_idx]
    # Normalize the intensity image
    scale = 255.0 / (robust_max_val - robust_min_val)
    normalized_intensity = scale * (intensity - robust_min_val)
    normalized_intensity = np.clip(normalized_intensity, 0, 255)
    return normalized_intensity


def visualize_intensity(intensity, robust=True, percentile=0.1) -> np.ndarray:
    """ Visualize intensity in grayscale [0-255] images. """
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.cpu().detach().numpy()
    if robust:
        intensity_image = robust_normalize(intensity, percentile)
    else:
        intensity_image = standard_normalize(intensity)
    intensity_image = intensity_image.astype(np.uint8)
    # intensity_image = Image.fromarray(intensity_image)
    return intensity_image


def rgb2gray(rgb: np.ndarray):
    """ Convert RGB to gray intensity values """
    # Using Formula Y601, the same as Matlab and OpenCV
    # https://en.wikipedia.org/wiki/Luma_%28video%29#Rec._601_luma_versus_Rec._709_luma_coefficients
    assert rgb.shape[-1] == 3, "The color channel is not the last dimension"
    # rgb.shape = [..., H, W, 3]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])  # [..., H, W, 3]
    gray = gray[..., None]  # [..., H, W, 1]
    return gray


def bgr2gray(bgr: np.ndarray):
    """ Convert RGB to gray intensity values """
    # Using Formula Y601, the same as Matlab and OpenCV
    # https://en.wikipedia.org/wiki/Luma_%28video%29#Rec._601_luma_versus_Rec._709_luma_coefficients
    assert bgr.shape[-1] == 3, "The color channel is not the last dimension"
    # rgb.shape = [..., H, W, 3]
    gray = np.dot(bgr[..., :], [0.114, 0.587, 0.299])  # [..., H, W, 3]
    gray = gray[..., None]  # [..., H, W, 1]
    return gray


def draw_warped_events(canvas: Image.Image, warped_events: np.ndarray) -> Image.Image:
    """ Draw warped events on the intensity image."""
    # List for positive and negative (warped) events
    pos_events = []
    neg_events = []
    num_events = warped_events.shape[0]
    for i in range(num_events):
        if warped_events[i, 2] > 0:
            pos_events.append(warped_events[i, 0])
            pos_events.append(warped_events[i, 1])
        else:
            neg_events.append(warped_events[i, 0])
            neg_events.append(warped_events[i, 1])

    # Plot warped event on the canvas
    color_canvas = canvas.convert('RGB')
    drawer = ImageDraw.Draw(color_canvas)
    drawer.point(pos_events, fill='red')
    drawer.point(neg_events, fill='blue')
    return color_canvas


def generate_event_mask(warped_events: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    """ Generate a mask for valid pixels. (1: has event, 0: no event) """
    num_events = warped_events.shape[0]
    warped_events = np.round(warped_events)
    # Initialize am empty np array
    mask = np.zeros((image_height, image_width), np.uint8)
    for i in range(num_events):
        x = int(warped_events[i, 0])
        y = int(warped_events[i, 1])
        mask[y, x] = 1
    return mask


def iwe2mask(iwe: np.ndarray) -> np.ndarray:
    """ Convert iwe to event mask."""
    mask = np.zeros((iwe.shape[0], iwe.shape[1]), dtype=np.uint8)
    valid_pixels = np.nonzero(iwe)
    mask[valid_pixels] = 1
    return mask


def mask_flow(flow_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """ Mask flow with the event mask """
    assert flow_image.height == mask.shape[0] and flow_image.width == mask.shape[1]
    masked_flow_image = flow_image.copy()
    for y in range(masked_flow_image.height):
        for x in range(masked_flow_image.width):
            if mask[y, x] == 0:
                # In case of invalid pixel, set it to 255 (white)
                masked_flow_image.putpixel((x, y), (255, 255, 255))
    return masked_flow_image


def semi_dense_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """ Use the mask to visualize semi-dense intensity images (only grayscale) """
    assert image.height == mask.shape[0] and image.width == mask.shape[1]
    for y in range(image.height):
        for x in range(image.width):
            if mask[y, x] == 0:
                # In case of invalid pixel, set it to 255 (white)
                image.putpixel((x, y), 255)
    return image


def visualize_iwe(iwe) -> np.ndarray:
    """ Visualize an IWE (H, W) """
    if isinstance(iwe, torch.Tensor):
        iwe = iwe.cpu().detach().numpy()
    # Normalize the IWE to [0, 255]
    max_val = iwe.max()
    min_val = iwe.min()
    scale = 255.0 / (max_val - min_val)
    iwe = scale * (iwe - min_val)
    # Invert color
    iwe = 255.0 - iwe
    # intensity_image = Image.fromarray(iwe.astype(np.uint8))
    return iwe.astype(np.uint8)


def visualize_iwe_batch(iwe_batch) -> np.ndarray:
    """ Visualize a batch of IWEs (b, 1, H, W) """
    if isinstance(iwe_batch, torch.Tensor):
        iwe = iwe_batch.cpu().detach().numpy()
    assert len(iwe_batch.shape) == 4
    iwe_list = []
    for i in range(iwe_batch.shape[0]):
        iwe = visualize_iwe(np.squeeze(iwe_batch[i, ...]))
        iwe_list.append(iwe)
    iwe_batch_disp = np.stack(iwe_list, axis=0)
    assert len(iwe_batch_disp.shape) == 3  # (b, H, W)
    return iwe_batch_disp