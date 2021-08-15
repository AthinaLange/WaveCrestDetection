"""
Wave Crests Detection

MIT License
(C) 2021 Athina Lange
"""

import os
import argparse
import logging
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from torchvision import transforms
from unet_resnet18 import UNet as UNet_resnet18
from tqdm import tqdm


def get_args():
    """ Get program arguments.

    Usages:
    - Help
      (pytorch)$: python train.py -h
    - Example
      (pytorch)$: python predict.py \
                  --model unet-resnet18 \
                  --load checkpoints/model_epoch350.pth \
                  --processing-window-bottom 4700 \
                  --image data/validate/images/20200707_Torrey_10C_1_timestack.png
    """
    parser = argparse.ArgumentParser(description='Wave Crests Detection.')

    parser.add_argument('--model', help='model',
                        choices=['unet-resnet18'],
                        default='unet-resnet18',
                        dest='model')
    parser.add_argument('--load', help='load model from a .pth file',
                        type=str, metavar='MODEL',
                        default=None,
                        dest='load', required=True)
    parser.add_argument('--sample-width', help='sample width',
                        type=int, metavar='N',
                        default=256,
                        dest='sample_width')
    parser.add_argument('--sample-height', help='sample height',
                        type=int, metavar='N',
                        default=256,
                        dest='sample_height')
    parser.add_argument('--image', help='load RGB image from a .png file',
                        type=str, metavar='IMAGE',
                        default=None,
                        dest='image', required=True)
    parser.add_argument('--processing-window-top', help='processing window top',
                        type=int, metavar='N',
                        default=0,
                        dest='processing_window_top')
    parser.add_argument('--processing-window-bottom', help='processing window bottom (-1 bottom of image)',
                        type=int, metavar='N',
                        default=-1,
                        dest='processing_window_bottom')
    parser.add_argument('--processing-window-left', help='processing window left',
                        type=int, metavar='N',
                        default=0,
                        dest='processing_window_left')
    parser.add_argument('--processing-window-right', help='processing window right (-1 right of image)',
                        type=int, metavar='N',
                        default=-1,
                        dest='processing_window_right')
    parser.add_argument('--processing-crop-margin', help='processing crop margin',
                        type=int, metavar='N',
                        default=16,
                        dest='processing_crop_margin')
    parser.add_argument('--filter-interactive', help='use filter interactively (or in batch mode)',
                        type=int, metavar='0/1',
                        default=1,
                        dest='filter_interactive_flag')
    parser.add_argument('--filter-threshold', help='filter threshold',
                        type=float, metavar='N',
                        default=0.05,
                        dest='filter_threshold')
    parser.add_argument('--filter-line-width', help='filter line width',
                        type=int, metavar='N',
                        default=3,
                        dest='filter_line_width')
    parser.add_argument('--filter-gap-close', help='filter gap close',
                        type=int, metavar='N',
                        default=5,
                        dest='filter_gap_close')
    parser.add_argument('--filter-length-min', help='filter length min',
                        type=int, metavar='N',
                        default=100,
                        dest='filter_length_min')
    parser.add_argument('--filter-skeleton', help='filter thin lines to skeleton',
                        type=int, metavar='0/1',
                        default=1,
                        dest='filter_skeleton_flag')
    parser.add_argument('--filter-surf-zone', help='filter restrict to surf zone',
                        type=int,metavar='0/1',
                        default=0,
                        dest='filter_surf_zone_flag')
    parser.add_argument('--filter-overlay-flag', help='filter overlay (on image + surf zone)',
                        type=int, metavar='0/1',
                        default=0,
                        dest='filter_overlay_flag')
    parser.add_argument('--prediction-suffix', help='prediction file suffix including file extension',
                        type=str, metavar='SUFFIX',
                        default='_prediction.jpg',
                        dest='prediction_suffix')
    parser.add_argument('--prediction-overlay-suffix', help='prediction overlay file suffix including file extension',
                        type=str, metavar='SUFFIX',
                        default='_prediction_overlay.png',
                        dest='prediction_overlay_suffix')

    return parser.parse_args()


def nothing(x):
    pass


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Program Arguments
    args = get_args()
    logging.info(f'''Program arguments:
      Model:                                         {args.model}
      Load:                                          {args.load}
      Sample width:                                  {args.sample_width}
      Sample height:                                 {args.sample_height}
      Image:                                         {args.image}
      Processing window top:                         {args.processing_window_top}
      Processing window bottom (-1 bottom of image): {args.processing_window_bottom}
      Processing window left:                        {args.processing_window_left}
      Processing window right (-1 right of image):   {args.processing_window_right}
      Processing crop margin:                        {args.processing_crop_margin}
      Filter interactive use (vs in batch mode)      {bool(args.filter_interactive_flag)} 
      Filter threshold:                              {args.filter_threshold}
      Filter line width:                             {args.filter_line_width}
      Filter gap close:                              {args.filter_gap_close}
      Filter length min:                             {args.filter_length_min}
      Filter skeleton:                               {bool(args.filter_skeleton_flag)}
      Filter surf zone:                              {bool(args.filter_surf_zone_flag)}
      Filter overlay:                                {bool(args.filter_overlay_flag)}
      Prediction file suffix:                        {args.prediction_suffix}
      Prediction overlay file suffix:                {args.prediction_overlay_suffix}''')

    # Device CPU vs CUDA(GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    logging.info(f'Device: {device}')

    # Read Model
    # ----------
    # Define model
    in_channels = 3  # RGB image
    out_channels = 1  # 1 Class and Background
    models = {"unet-resnet18": UNet_resnet18()}
    model = models[args.model]
    # Load model
    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    # Determine prediction size
    inputs = torch.ones(1, in_channels, args.sample_height, args.sample_width)
    outputs = model(inputs)
    prediction_height = outputs.shape[2]
    prediction_width = outputs.shape[3]

    # Read Image
    # ----------
    image_org = cv2.imread(args.image)
    # Pytorch color images are RGB, convert Opencv format BRG
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    # Transpose HWC of image to CHW for pytorch
    # Normalize 8-bit image data to [0.0, 1.0)
    tfs = transforms.Compose([
        transforms.ToTensor()
    ])
    image = tfs(image)
    image = image.to(device=device, dtype=torch.float32)

    # Tile-based Processing
    # ---------------------
    model.eval()
    # Prediction image
    prediction = torch.ones(out_channels, image.shape[1], image.shape[2])
    prediction_tile = torch.zeros(prediction_height, prediction_width)
    # Processing window
    # - rows
    processing_window_top = args.processing_window_top
    processing_window_bottom = image.shape[1] - 1
    if 0 < args.processing_window_bottom:
        processing_window_bottom = args.processing_window_bottom
    processing_nb_rows = processing_window_bottom - processing_window_top + 1
    # - columns
    processing_window_left = args.processing_window_left
    processing_window_right = image.shape[2] - 1
    if 0 < args.processing_window_right:
        processing_window_right = args.processing_window_right
    processing_nb_columns = processing_window_right - processing_window_left + 1
    # Adjust offset if crop margin or prediction is smaller than ground-truth
    x_offset_out = (args.sample_width - prediction_width) // 2 + args.processing_crop_margin
    y_offset_out = (args.sample_height - prediction_height) // 2 + args.processing_crop_margin
    crop_width = prediction_width - 2 * args.processing_crop_margin
    crop_height = prediction_height - 2 * args.processing_crop_margin
    # Show progress bar
    nb_tiles = ((processing_nb_rows - args.sample_height) // crop_height + 2) * \
               ((processing_nb_columns - args.sample_width) // crop_width + 2)
    with tqdm(total=nb_tiles, desc='Tile-based Prediction', unit='tile') as progress_bar:
        # Go through rows
        for y_in in range(processing_window_bottom - args.sample_height, processing_window_top, -crop_height):
            y_out = y_in + y_offset_out
            # Go through columns
            j = 0
            for x_in in range(processing_window_left, processing_window_right - args.sample_width, crop_width):
                x_out = x_in + x_offset_out
                # Extract image tile
                image_tile = image[:, y_in:y_in + args.sample_height, x_in:x_in + args.sample_width]
                image_tile = image_tile[np.newaxis, ...]  # Batch Size = 1 - Channels - Height - Width
                # Run the prediction model on a tile
                with torch.no_grad():
                    prediction_tile = model(image_tile)
                # Insert prediction tile into prediction image
                prediction[0, y_out:y_out + crop_height, x_out:x_out + crop_width] = \
                    prediction_tile[0, 0,
                                    args.processing_crop_margin:args.processing_crop_margin + crop_height,
                                    args.processing_crop_margin:args.processing_crop_margin + crop_width]
                progress_bar.update(1)
            # Last column to adjust to any processing window width
            x_in = processing_window_right - args.sample_width - 1
            x_out = x_in + x_offset_out
            image_tile = image[:, y_in:y_in + args.sample_height, x_in:x_in + args.sample_width]
            image_tile = image_tile[np.newaxis, ...]
            with torch.no_grad():
                prediction_tile = model(image_tile)
            prediction[0, y_out:y_out + crop_height, x_out:x_out + crop_width] = \
                prediction_tile[0, 0,
                                args.processing_crop_margin:args.processing_crop_margin + crop_height,
                                args.processing_crop_margin:args.processing_crop_margin + crop_width]
            progress_bar.update(1)
        # Last row to adjust to any processing window height
        y_in = processing_window_top
        y_out = y_in + y_offset_out
        # Go through columns
        for x_in in range(processing_window_left, processing_window_right - args.sample_width, crop_width):
            x_out = x_in + x_offset_out
            image_tile = image[:, y_in:y_in + args.sample_height, x_in:x_in + args.sample_width]
            image_tile = image_tile[np.newaxis, ...]
            with torch.no_grad():
                prediction_tile = model(image_tile)
            prediction[0, y_out:y_out + crop_height, x_out:x_out + crop_width] = \
                prediction_tile[0, 0,
                                args.processing_crop_margin:args.processing_crop_margin + crop_height,
                                args.processing_crop_margin:args.processing_crop_margin + crop_width]
            progress_bar.update(1)
        # Last column
        x_in = processing_window_right - args.sample_width - 1
        x_out = x_in + x_offset_out
        image_tile = image[:, y_in:y_in + args.sample_height, x_in:x_in + args.sample_width]
        image_tile = image_tile[np.newaxis, ...]
        with torch.no_grad():
            prediction_tile = model(image_tile)
        prediction[0, y_out:y_out + crop_height, x_out:x_out + crop_width] = \
            prediction_tile[0, 0,
                            args.processing_crop_margin:args.processing_crop_margin + crop_height,
                            args.processing_crop_margin:args.processing_crop_margin + crop_width]
        progress_bar.update(1)

    # Interactive Filter (using OpenCV)
    # ---------------------------------
    if bool(args.filter_interactive_flag):
        logging.info("Adjust parameters interactively and hit ENTER to exit")
    # Convert to numpy array
    image = image.detach().cpu()
    image = image.numpy().transpose((1, 2, 0))  # CHW to HWC
    prediction = torch.sigmoid(prediction)
    prediction = prediction.detach().cpu()
    prediction = prediction.numpy().transpose((1, 2, 0))
    if bool(args.filter_interactive_flag):
        # Create display
        cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
        # Create trackbars
        cv2.createTrackbar('Threshold /100', 'Prediction', int(args.filter_threshold * 100), 100, nothing)
        cv2.createTrackbar('Line Width odd values only', 'Prediction', args.filter_line_width, 15, nothing)
        cv2.setTrackbarMin('Line Width odd values only', 'Prediction', 3)
        cv2.createTrackbar('Gap Close odd values only', 'Prediction', args.filter_gap_close, 31, nothing)
        cv2.createTrackbar('Min Length', 'Prediction', args.filter_length_min, image.shape[0], nothing)
        cv2.createTrackbar('Skeleton Off(0)/On(1)', 'Prediction', bool(args.filter_skeleton_flag), 1, nothing)
        cv2.createTrackbar('Surf Zone Off(0)/On(1)', 'Prediction', bool(args.filter_surf_zone_flag), 1, nothing)
        cv2.createTrackbar('Overlay Off(0)/On(1)', 'Prediction', bool(args.filter_overlay_flag), 1, nothing)
    # Surf Zone
    # - smooth
    image_blurred = cv2.GaussianBlur(np.uint8(255 * image[:, :, 0]), (31, 31), 0)
    # - threshold
    _, surf_zone = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    surf_zone_floodfill = surf_zone.copy()
    # - flood from top
    h, w = surf_zone_floodfill.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(surf_zone_floodfill, mask, (0, 0), 255)
    surf_zone_floodfill_inv = cv2.bitwise_not(surf_zone_floodfill)
    surf_zone = surf_zone | surf_zone_floodfill_inv
    # - eliminate small regions
    nb_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(np.uint8(255*surf_zone), connectivity=8)
    for label in range(nb_labels):
        if stats[label, cv2.CC_STAT_AREA] < h * 100:
            surf_zone[labels == label] = 0.0
    # Update logic
    # make sure that filters are executed in sequence
    update = True
    threshold_old = None
    line_width_old = None
    gap_close_old = None
    length_min_old = None
    skeleton_flag_old = None
    surf_zone_flag_old = None
    overlay_flag_old = None
    prediction_mask_threshold = None
    prediction_mask_peak = None
    prediction_mask_gaps = None
    prediction_mask_length = None
    prediction_mask_skeleton = None
    prediction_mask = None
    while True:
        if bool(args.filter_interactive_flag):
            # Get current positions of trackbars
            threshold = cv2.getTrackbarPos('Threshold /100', 'Prediction')/100.0
            line_width = cv2.getTrackbarPos('Line Width odd values only', 'Prediction')
            cv2.setTrackbarPos('Line Width odd values only', 'Prediction', (line_width // 2) * 2 + 1)
            line_width = cv2.getTrackbarPos('Line Width odd values only', 'Prediction')
            gap_close = cv2.getTrackbarPos('Gap Close odd values only', 'Prediction')
            cv2.setTrackbarPos('Gap Close odd values only', 'Prediction', (gap_close // 2) * 2 + 1)
            gap_close = cv2.getTrackbarPos('Gap Close odd values only', 'Prediction')
            length_min = cv2.getTrackbarPos('Min Length', 'Prediction')
            skeleton_flag = cv2.getTrackbarPos('Skeleton Off(0)/On(1)', 'Prediction')
            surf_zone_flag = cv2.getTrackbarPos('Surf Zone Off(0)/On(1)', 'Prediction')
            overlay_flag = cv2.getTrackbarPos('Overlay Off(0)/On(1)', 'Prediction')
        else:
            threshold = args.filter_threshold
            line_width = args.filter_line_width
            gap_close = args.filter_gap_close
            length_min = args.filter_length_min
            skeleton_flag = args.filter_skeleton_flag
            surf_zone_flag = args.filter_surf_zone_flag
            overlay_flag = args.filter_overlay_flag

        # Threshold
        if threshold != threshold_old or update:
            logging.info(f"Threshold {threshold}")
            threshold_old = threshold
            update = True
            prediction_mask_threshold = prediction.copy()
            prediction_mask_threshold = prediction_mask_threshold * (prediction_mask_threshold > threshold)

        # Detect peaks in lines
        if line_width != line_width_old or update:
            logging.info(f"Detect peaks in lines {line_width}")
            line_width_old = line_width
            update = True
            prediction_mask_peak = prediction_mask_threshold.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1))
            prediction_mask_peak = cv2.morphologyEx(prediction_mask_peak, cv2.MORPH_TOPHAT, kernel)
            prediction_mask_peak = 1.0 * (prediction_mask_peak > 0)

        # Close small gaps in line
        if gap_close != gap_close_old or update:
            logging.info(f"Close small gaps in line {gap_close}")
            gap_close_old = gap_close
            update = True
            prediction_mask_gaps = prediction_mask_peak.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, gap_close))
            prediction_mask_gaps = cv2.morphologyEx(prediction_mask_gaps, cv2.MORPH_CLOSE, kernel)

        # Min length
        if length_min != length_min_old or update:
            logging.info(f"Min length {length_min}")
            length_min_old = length_min
            update = True
            prediction_mask_length = prediction_mask_gaps.copy()
            nb_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(np.uint8(255 * prediction_mask_length), connectivity=8)
            for label in range(nb_labels):
                if stats[label, cv2.CC_STAT_HEIGHT] < length_min or \
                        stats[label, cv2.CC_STAT_AREA] < length_min:
                    prediction_mask_length[labels == label] = 0.0

        # Skeleton
        if skeleton_flag != skeleton_flag_old or update:
            logging.info(f"Skeleton {bool(skeleton_flag)}")
            skeleton_flag_old = skeleton_flag
            update = True
            prediction_mask_skeleton = prediction_mask_length.copy()
            skeleton = prediction_mask_skeleton.copy()
            skeleton[:, :] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            while True:
                eroded = cv2.morphologyEx(prediction_mask_skeleton, cv2.MORPH_ERODE, kernel)
                temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
                temp = cv2.subtract(prediction_mask_skeleton, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                prediction_mask_skeleton[:, :] = eroded[:, :]
                if cv2.countNonZero(prediction_mask_skeleton) == 0:
                    break
            prediction_mask_skeleton = skeleton

        # Surf Zone
        if surf_zone_flag != surf_zone_flag_old or update:
            logging.info(f"Surf Zone {bool(surf_zone_flag)}")
            surf_zone_flag_old = surf_zone_flag
            update = True
            prediction_mask = prediction_mask_skeleton.copy()
            if bool(surf_zone_flag):
                prediction_mask[surf_zone < 255] = 0.0
            prediction_mask = 1.0 - (prediction_mask > 0)

        if bool(args.filter_interactive_flag):
            if overlay_flag != overlay_flag_old or update:
                logging.info(f"Overlay {bool(overlay_flag)}")
                overlay_flag_old = overlay_flag
                update = True
                if bool(overlay_flag):
                    overlay = image.copy()
                    # Create contour of surf zone
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    surf_zone_contour = cv2.morphologyEx(surf_zone, cv2.MORPH_GRADIENT, kernel)
                    # Convert RGB to BRG and overlay prediction in red and surf zone contour in black
                    overlay[:, :, 2] = np.maximum(np.minimum(image[:, :, 0], 1.0 - surf_zone_contour), 1.0 - prediction_mask)
                    overlay[:, :, 1] = np.minimum(np.minimum(image[:, :, 1], 1.0 - surf_zone_contour), prediction_mask)
                    overlay[:, :, 0] = np.minimum(np.minimum(image[:, :, 2], 1.0 - surf_zone_contour), prediction_mask)
                    cv2.imshow("Prediction", overlay)
                else:
                    cv2.imshow("Prediction", prediction_mask)

        if update:
            logging.info("---")
        update = False

        if not bool(args.filter_interactive_flag):
            break

        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # ENTER
            break
    cv2.destroyAllWindows()

    # Save Prediction
    filename = os.path.splitext(os.path.basename(args.image))[0] + args.prediction_suffix
    cv2.imwrite(filename, 255 * prediction_mask)
    logging.info(f"Saved prediction image as {filename}")
    # Overlay
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    surf_zone_contour = cv2.morphologyEx(surf_zone, cv2.MORPH_GRADIENT, kernel)
    surf_zone_contour = 0 < surf_zone_contour
    overlay = image_org.copy()
    overlay[:, :, 0] = np.minimum(overlay[:, :, 0], 255 * (1.0 - surf_zone_contour))
    overlay[:, :, 1] = np.minimum(overlay[:, :, 1], 255 * (1.0 - surf_zone_contour))
    overlay[:, :, 2] = np.minimum(overlay[:, :, 2], 255 * (1.0 - surf_zone_contour))
    overlay[:, :, 0] = np.minimum(overlay[:, :, 0], 255 * prediction_mask)
    overlay[:, :, 1] = np.minimum(overlay[:, :, 1], 255 * prediction_mask)
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], 255 * (1.0 - prediction_mask))
    filename = os.path.splitext(os.path.basename(args.image))[0] + args.prediction_overlay_suffix
    cv2.imwrite(filename, overlay)
    logging.info(f"Saved prediction overlay image as {filename}")

    if bool(args.filter_interactive_flag):
        # Plot (using Matplotlib)
        # -----------------------
        logging.info("Close Figure to end program")
        fig, axs = plt.subplots(4, 1)
        fig.set_size_inches(5, 10)
        image_ax = axs[0]
        prediction_overlay_ax = axs[1]
        prediction_mask_ax = axs[2]
        prediction_ax = axs[3]
        # Plot Image
        image_ax.imshow(image)
        image_ax.set_xticks([0, image.shape[1] - 1])
        image_ax.set_xticklabels([0, image.shape[1] - 1])
        image_ax.set_yticks([0, image.shape[0] - 1])
        image_ax.set_yticklabels([0, image.shape[0] - 1])
        image_ax.set_title("Image", fontsize=12)
        # Plot Image with Processed Prediction Overlay
        prediction_overlay_ax.imshow(image)
        prediction_overlay_ax.imshow(1.0 - prediction_mask, cmap="Reds", alpha=0.5)
        prediction_overlay_ax.set_title("Prediction", fontsize=12)
        prediction_overlay_ax.axis("off")
        # Plot Processed Prediction
        prediction_mask_ax.imshow(1.0 - prediction_mask, cmap="Reds")
        prediction_mask_ax.axis("off")
        # Plot Prediction
        prediction_ax.imshow(prediction, cmap="Reds")
        prediction_ax.axis("off")
        plt.tight_layout()
        plt.show()
