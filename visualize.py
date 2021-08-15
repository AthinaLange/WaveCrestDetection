"""
Wave Crests Time Stack Sample Visualizer

MIT License
(C) 2021 Athina Lange
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_sample(image, meta, ground_truth, ground_truth_mask=None, prediction=None):

    if prediction is None:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(9, 5)
        image_ax = axs[0]
        ground_truth_ax = axs[1]
    else:
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(9, 9)
        image_ax = axs[0, 0]
        ground_truth_ax = axs[0, 1]
        prediction_ax = axs[1, 0]
        prediction_overlay_ax = axs[1, 1]

    # Plot image
    image = image.detach().cpu()
    image = image.numpy().transpose((1, 2, 0))  # CHW to HWC
    image_ax.imshow(image)
    image_ax.set_xticks([0, image.shape[1]-1])
    image_ax.set_xticklabels([meta["x start"], meta["x end"]-1])
    image_ax.set_yticks([0, image.shape[0]-1])
    image_ax.set_yticklabels([meta["y start"], meta["y end"]-1])
    image_ax.set_title("Image", fontsize=12)

    # Plot ground-truth as overlay on image
    ground_truth = ground_truth.detach().cpu()
    ground_truth = ground_truth.numpy().transpose((1, 2, 0))
    ground_truth_ax.imshow(image)
    if ground_truth_mask is not None:
        ground_truth_mask = ground_truth_mask.detach().cpu()
        ground_truth_mask = ground_truth_mask.numpy().transpose((1, 2, 0))  # CHW to HWC
        ground_truth_ax.imshow(ground_truth_mask, cmap="Greens", alpha=0.5)
    ground_truth_ax.imshow(ground_truth, cmap="Reds", alpha=0.5)
    ground_truth_ax.set_title("Ground-Truth", fontsize=12)
    ground_truth_ax.axis("off")

    # Plot prediction
    if prediction is not None:
        prediction = prediction.detach().cpu()
        prediction = prediction.numpy().transpose((1, 2, 0))

        # Pad prediction (if prediction is smaller than ground-truth)
        if prediction.shape[0] < ground_truth.shape[0] or \
           prediction.shape[1] < ground_truth.shape[1]:
            y = int((ground_truth.shape[0] - prediction.shape[0]) / 2)
            x = int((ground_truth.shape[1] - prediction.shape[1]) / 2)
            image_padded = np.zeros(image.shape)
            image_padded[y:y + prediction.shape[0], x:x + prediction.shape[1], :] = \
                image[y:y + prediction.shape[0], x:x + prediction.shape[1], :]
            prediction_padded = np.zeros(ground_truth.shape)
            prediction_padded[y:y + prediction.shape[0], x:x + prediction.shape[1], :] = \
                prediction
            image = image_padded
            prediction = prediction_padded

        prediction_ax.imshow(prediction, cmap="Reds")
        prediction_ax.set_title("Prediction", fontsize=12)
        prediction_ax.axis("off")

        # Overlay on image
        # Detect peaks in lines
        prediction = prediction * (prediction > 0.1)
        prediction_mask = cv2.morphologyEx(prediction,
                                           cv2.MORPH_TOPHAT,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
        prediction_overlay_ax.imshow(image)
        prediction_overlay_ax.imshow(prediction_mask > 0.0, cmap="Reds", alpha=0.5)
        prediction_overlay_ax.set_title("Prediction", fontsize=12)
        prediction_overlay_ax.axis("off")

    # Title
    fig.suptitle(meta["image id"], fontsize=14)

    return fig
