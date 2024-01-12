# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torchvision
from pathlib import Path

from monai.data import DataLoader, Dataset

from monai.transforms import (
    RandSpatialCropSamples,
    LoadImage,
    Compose,
    AddChannel,
    RepeatChannel,
    ToTensor,
    Transform,
    Transpose,
    CastToType,
    EnsureType,
    ScaleIntensityRangePercentiles,
)
from tqdm.contrib import tenumerate
from aicsimageio import AICSImage

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from compressai.zoo.pretrained import load_pretrained

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def init_weights(net, init_type="kaiming", init_gain=0.002):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method:
                           normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming
    might work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight"):
            if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm") != -1:
                # BatchNorm Layer's weight is not a matrix; only normal distribution.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class LoadTiff(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        print(data)
        x = AICSImage(data)
        img = x.get_image_data("YX", S=0, T=0, C=0)
        img = img.astype(np.float32)
        return img
    
class Normalize(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        # Rescale unint16 values to [0,1]
        result = img / 65535.0
        return result
        
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, use_3D
):
    model.train()
    device = next(model.parameters()).device
    metric = 'mse' if str(criterion.metric) == 'MSELoss()' else 'ms_ssim' 

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        # add the image comparison: input and the prediction
        input_img = d.detach().cpu()
        predicted_img = out_net['x_hat'].detach().cpu()
        if use_3D:
            input_slice = input_img[0].squeeze(0)[input_img.shape[2] // 2].unsqueeze(0) # C x H x W
            predicted_slice = predicted_img[0].squeeze(0)[predicted_img.shape[2] // 2].unsqueeze(0) # C x H x W
            comparison = torchvision.utils.make_grid([input_slice, predicted_slice])
        else:
            comparison = torchvision.utils.make_grid([input_img[0], predicted_img[0]])
        # Write losses to TensorBoard
        writer.add_image('image_comparison', comparison, global_step=epoch)
        writer.add_scalar(f'Loss/{metric}_loss', out_criterion[f"{metric}_loss"].item(), global_step=epoch)
        writer.add_scalar('Loss/bpp_loss', out_criterion["bpp_loss"].item(), global_step=epoch)
        writer.add_scalar('Loss/total_loss', out_criterion["loss"].item(), global_step=epoch)
        writer.add_scalar('Loss/aux_loss', aux_loss.item(), global_step=epoch)
        
        if i % 5 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\t{metric} loss: {out_criterion[f"{metric}_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    metric = 'mse' if str(criterion.metric) == 'MSELoss()' else 'ms_ssim' 
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    metric_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            metric_loss.update(out_criterion[f"{metric}_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\t{metric} loss: {metric_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if not isinstance(filename,Path):
        filename = Path(filename)
    torch.save(state, filename)
    if is_best:
        best_filename = filename.parent / 'best.pth.tar'
        shutil.copyfile(filename, best_filename)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        default=3,
        type=int,
        help="Model quality (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        default='mse',
        choices=['mse','ms-ssim'],
        help="Model metric (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_checkpoint.pth.tar", help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--train_split", type=str, default = "train", help="split for train dataset")
    parser.add_argument("--test_split", type=str, default = "test", help="split for test dataset")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model or not")
    parser.add_argument("--use_3D", action="store_true", help="Use 3D model or not")
    
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    assert not args.use_3D ^ ('3d' in args.model), "if use 3d, hand in 3d model, please keep it consistent."
    if args.use_3D:
        train_transforms = Compose(
        [LoadImage(image_only=True),AddChannel(), Transpose(indices = (0,3,1,2)), Normalize(), RandSpatialCropSamples(roi_size = (32,256,256), num_samples = 4, random_size = False)]
    )

        test_transforms = Compose(
            [LoadImage(image_only=True),AddChannel(), Transpose(indices = (0,3,1,2)), Normalize(), RandSpatialCropSamples(roi_size = (32,256,256), num_samples = 1, random_size = False, random_center = False)]
        )
    else:
        train_transforms = Compose(
            [LoadImage(image_only=True),AddChannel(),Normalize(),RandSpatialCropSamples(roi_size = (256,256), num_samples = 4, random_size = False)]
        )

        test_transforms = Compose(
            [LoadImage(image_only=True),AddChannel(), Normalize(),RandSpatialCropSamples(roi_size = (256,256), num_samples = 1, random_size = False, random_center = False)]
        )

    train_dataset = ImageFolder(args.dataset, split=args.train_split, transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split=args.test_split, transform=test_transforms)
    print(f"------------length of training dataset is {len(train_dataset)}.")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    net = image_models[args.model](quality=args.quality, pretrained=args.pretrained, metric = args.metric)
    
    #initialization:
    if not args.pretrained: # not use author provided.
        init_weights(net, init_type = "kaiming")
    if args.checkpoint:  # but load from our own previous checkpoint
        print("Loading", args.checkpoint)
        state_dict = torch.load(args.checkpoint)['state_dict']
        state_dict = load_pretrained(state_dict)
        net = net.from_state_dict(state_dict)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric = args.metric)

    last_epoch = 0

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args.use_3D
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {   "quality": args.quality,
                "metric": args.metric,
                "model": args.model,
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
            filename = args.save_path
        )


if __name__ == "__main__":
    main(sys.argv[1:])
