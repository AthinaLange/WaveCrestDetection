"""
Wave Crests Detection Model Training

The training module can be used as a library and a stand-alone program.

MIT License
(C) 2021 Athina Lange
"""

import sys
import os
import argparse
import logging
import warnings

import torch
import torch.nn as nn
import kornia
warnings.filterwarnings("ignore", category=UserWarning)
from torch import optim
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from tqdm import tqdm

from unet_resnet18 import UNet as UNet_resnet18
from loss import Loss as Loss
from dataset import WaveCrestsTimeStackDataset
from visualize import plot_sample


def validate(model,
             criterion,
             validate_dataloader,
             mask_width,
             epoch,
             writer,
             device):
    """Validate the model.
    """

    model.eval()

    nb_validate_batches = len(validate_dataloader)
    total_loss, n = 0, 0

    # Show progress bar
    with tqdm(total=nb_validate_batches, desc='Validate ', unit='batch') as progress_bar:

        # Get batch of validation data
        for batch_idx, batch in enumerate(validate_dataloader):
            image_batch = batch['image']
            ground_truth_batch = batch['ground-truth']
            meta_batch = batch["meta"]

            # Check dimensions
            assert image_batch.shape[1] == model.in_channels, \
                f'Validate: miss-match in channels between model: ' \
                f'{model.in_channels} and data: {image_batch.shape[1]}'
            assert ground_truth_batch.shape[1] == model.out_channels, \
                f'Validate: Miss-match out channels between model: ' \
                f'{model.out_channels} and data: {ground_truth_batch.shape[1]}'
            # Adjust data type
            image_batch = image_batch.to(
                device=device, dtype=torch.float32)
            ground_truth_batch = ground_truth_batch.to(
                device=device, dtype=(torch.float32 if model.out_channels == 1 else torch.long))

            # When the ground-truth is not provided for all objects in the image
            # limit loss function to regions around ground-truth
            ground_truth_mask_batch = None
            if 0 < mask_width:
                ground_truth_mask_batch = kornia.morphology.dilation(
                    ground_truth_batch, torch.ones(1, mask_width).to(device=device))
                if 0.0 == ground_truth_mask_batch.sum():
                    continue

            # Prediction
            with torch.no_grad():
                prediction_batch = model(image_batch)
            # Loss
            loss = criterion(prediction_batch, ground_truth_batch, ground_truth_mask_batch)
            total_loss += loss.item()
            n += 1

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(**{'loss-validate': total_loss / n})

            # Update tensorboard logging
            if batch_idx < 2:
                for idx in range(image_batch.shape[0]):
                    image = image_batch[idx]
                    ground_truth = ground_truth_batch[idx]
                    ground_truth_mask = None
                    if ground_truth_mask_batch is not None:
                        ground_truth_mask = ground_truth_mask_batch[idx]
                    prediction = prediction_batch[idx]
                    meta = {
                        "idx": meta_batch["sample idx"][idx].item(),
                        "image id": meta_batch["image id"][idx],
                        "x start": meta_batch["x start"][idx].item(),
                        "x end": meta_batch["x end"][idx].item(),
                        "y start": meta_batch["y start"][idx].item(),
                        "y end": meta_batch["y end"][idx].item()
                    }
                    fig = plot_sample(image=image,
                                      meta=meta,
                                      ground_truth=ground_truth,
                                      ground_truth_mask=ground_truth_mask,
                                      prediction=torch.sigmoid(prediction))
                    writer.add_figure("validate/batch{}/{}".format(batch_idx, idx), fig, epoch + 1)

    if 0 < n:
        total_loss = total_loss / n

    return total_loss


def train(model,
          criterion,
          optimizer,
          scheduler,
          train_dataloader,
          validate_dataloader,
          mask_width,
          epochs,
          writer,
          device='cpu'):
    """ Train the model.
    """

    nb_train_batches = len(train_dataloader)
    for epoch in range(epochs):
        model.train()
        epoch_loss, n = 0, 0

        # Show progress bar
        with tqdm(total=nb_train_batches, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as progress_bar:

            # Get batch of training data
            for batch_idx, batch in enumerate(train_dataloader):
                image_batch = batch['image']
                ground_truth_batch = batch['ground-truth']
                meta_batch = batch["meta"]

                # Check dimensions
                assert image_batch.shape[1] == model.in_channels, \
                    f'Train: Miss-match in channels between model: ' \
                    f'{model.in_channels} and data: {image_batch.shape[1]}'
                assert ground_truth_batch.shape[1] == model.out_channels, \
                    f'Train: Miss-match out channels between model: ' \
                    f'{model.out_channels} and data: {ground_truth_batch.shape[1]}'
                # Adjust data type
                image_batch = image_batch.to(
                    device=device, dtype=torch.float32)
                ground_truth_batch = ground_truth_batch.to(
                    device=device, dtype=(torch.float32 if model.out_channels == 1 else torch.long))

                # When the ground-truth is not provided for all objects in the image
                # limit loss function to regions around ground-truth
                ground_truth_mask_batch = None
                if 0 < mask_width:
                    ground_truth_mask_batch = kornia.morphology.dilation(
                        ground_truth_batch, torch.ones(1, mask_width).to(device=device))
                    if 0.0 == ground_truth_mask_batch.sum():
                        continue

                # Prediction
                prediction_batch = model(image_batch)
                # Loss
                loss = criterion(prediction_batch, ground_truth_batch, ground_truth_mask_batch)
                epoch_loss += loss.item()
                n += 1
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                # + gradient clipping
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(**{'loss-train': epoch_loss / n})

                # Update tensorboard logging
                if batch_idx == nb_train_batches - 1:  # use before last batch
                    for idx in range(image_batch.shape[0]):
                        image = image_batch[idx]
                        ground_truth = ground_truth_batch[idx]
                        ground_truth_mask = None
                        if ground_truth_mask_batch is not None:
                            ground_truth_mask = ground_truth_mask_batch[idx]
                        prediction = prediction_batch[idx]
                        meta = {
                            "idx": meta_batch["sample idx"][idx].item(),
                            "image id": meta_batch["image id"][idx],
                            "x start": meta_batch["x start"][idx].item(),
                            "x end": meta_batch["x end"][idx].item(),
                            "y start": meta_batch["y start"][idx].item(),
                            "y end": meta_batch["y end"][idx].item()
                        }
                        fig = plot_sample(image=image,
                                          meta=meta,
                                          ground_truth=ground_truth,
                                          ground_truth_mask=ground_truth_mask,
                                          prediction=torch.sigmoid(prediction))
                        writer.add_figure("train/batch{}/{}".format(batch_idx, idx), fig, epoch + 1)

        if 0 < n:
            epoch_loss = epoch_loss / n
        # Validate the Model
        validation_loss = validate(model=model,
                                   criterion=criterion,
                                   validate_dataloader=validate_dataloader,
                                   mask_width=mask_width,
                                   epoch=epoch,
                                   writer=writer,
                                   device=device)

        # Reduce learning rate when the validation loss has stopped improving
        if args.scheduler == "ReduceLROnPlateau":
            scheduler.step(validation_loss)
        # Reduce learning rate at constant intervals
        if args.scheduler == "StepLR":
            scheduler.step()

        # Update tensorboard logging
        writer.add_scalar('learning-rate', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalars('loss', {"train": epoch_loss,
                                    "validate": validation_loss}, epoch + 1)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch + 1)
            # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch + 1)

        # Save model at checkpoints
        try:
            os.mkdir('checkpoints/')
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(model.state_dict(),
                   f'checkpoints/model_epoch{epoch + 1}.pth')
        logging.info(f'Saved model at checkpoint epoch {epoch + 1}.')

    writer.close()


def get_args():
    """ Get program arguments.

    Usages:
    - Help
      (pytorch)$: python train.py -h
    - Logging with a copy of stdout and stderr to file
      (pytorch)$: python train.py 2>&1 | tee train.log
      train.log
    """
    parser = argparse.ArgumentParser(description='Train Wave Crests Detection model.')

    parser.add_argument('--model', help='model',
                        choices=['unet-resnet18'],
                        default='unet-resnet18',
                        dest='model')
    parser.add_argument('--loss', help='loss function',
                        choices=['BCE'],
                        default='BCE',
                        dest='loss')
    parser.add_argument('--optimizer', help='optimizer',
                        choices=['RMS', 'Adam'],
                        default='RMS',
                        dest='optimizer')
    parser.add_argument('--scheduler', help='scheduler',
                        choices=['ReduceLROnPlateau', 'StepLR'],
                        default='ReduceLROnPlateau',
                        dest='scheduler')
    parser.add_argument('--samples-per-image', help='number of samples per image',
                        type=int, metavar='N',
                        default=1000,
                        dest='samples_per_image')
    parser.add_argument('--sample-width', help='sample width',
                        type=int, metavar='N',
                        default=256,
                        dest='sample_width')
    parser.add_argument('--sample-height', help='sample height',
                        type=int, metavar='N',
                        default=256,
                        dest='sample_height')
    parser.add_argument('--mask-width', help='ground-truth mask width (0 for no mask), dilation in x',
                        type=int, metavar='N',
                        default=100,
                        dest='mask_width')
    parser.add_argument('--epochs', help='number of epochs',
                        type=int, metavar='N',
                        default=50,
                        dest='epochs')
    parser.add_argument('--batch-size', help='batch size',
                        type=int, metavar='N',
                        default=8,
                        dest='batch_size')
    parser.add_argument('--learning-rate', help='learning rate',
                        type=float, metavar='N',
                        default=0.0001,
                        dest='learning_rate')
    parser.add_argument('--load', help='load model from a .pth file',
                        type=str, metavar='MODEL',
                        default=None,
                        dest='load')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Program Arguments
    args = get_args()
    logging.info(f'''Program arguments:
      Model:             {args.model}
      Loss:              {args.loss}
      Optimizer:         {args.optimizer}
      Scheduler:         {args.scheduler}
      Samples per image: {args.samples_per_image}
      Sample width:      {args.sample_width}
      Sample width:      {args.sample_height}
      Mask width:        {args.mask_width}
      Epochs:            {args.epochs}
      Batch size:        {args.batch_size}
      Learning rate:     {args.learning_rate}
      Load:              {args.load}''')

    # Device CPU vs CUDA(GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    # Model
    in_channels = 3  # RGB image
    out_channels = 1  # 1 Class and Background
    models = {"unet-resnet18": UNet_resnet18()}
    model = models[args.model]
    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    inputs = torch.ones(args.batch_size, in_channels, args.sample_height, args.sample_width)
    logging.info(f'Model: {args.model}\n{summary(model, inputs.to(device))}')

    # Loss (criterion)
    criterion = Loss(args.loss)
    # Optimizer
    optimizers = {"RMS":  optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-8, momentum=0.9),
                  "Adam": optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)}
    optimizer = optimizers[args.optimizer]
    # Scheduler
    schedulers = {"ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10),
                  "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)}
    scheduler = schedulers[args.scheduler]

    # Dataset and Dataset Loader
    datasets = {
        "train": WaveCrestsTimeStackDataset(
            images_dir="data/train/images",
            ground_truth_dir="data/train/ground_truth",
            ground_truth_suffix="_ground_truth",
            samples_per_image=args.samples_per_image,
            sample_width=args.sample_width,
            sample_height=args.sample_height),
        "validate": WaveCrestsTimeStackDataset(
            images_dir="data/validate/images",
            ground_truth_dir="data/validate/ground_truth",
            ground_truth_suffix="_ground_truth",
            samples_per_image=args.samples_per_image,
            sample_width=args.sample_width,
            sample_height=args.sample_height)
    }
    dataloaders = {
        "train": DataLoader(datasets["train"],
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8),
        "validate": DataLoader(datasets["validate"],
                               batch_size=1,
                               shuffle=False,
                               num_workers=8)
    }

    try:

        # Logging data for tensorflow is stored under ./runs
        writer = SummaryWriter(comment=f'_Epochs_{args.epochs}' \
                                       f'_BatchSize_{args.batch_size}' \
                                       f'_LearningRate_{args.learning_rate}')

        # Train the Model
        train(model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              train_dataloader=dataloaders["train"],
              validate_dataloader=dataloaders["validate"],
              mask_width=args.mask_width,
              epochs=args.epochs,
              writer=writer,
              device=device)

    except KeyboardInterrupt:
        # Save the current model when interrupted
        torch.save(model.state_dict(), 'model_interrupted.pth')
        logging.info('Saved model when interrupted.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
