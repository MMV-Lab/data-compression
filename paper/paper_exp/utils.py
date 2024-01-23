import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Sequence
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import monai
from monai.transforms import NormalizeIntensity
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_list = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.val_list.append(self.val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def std(self):
        return np.std(np.array(self.val_list))

    @property
    def var(self):
        return np.var(np.array(self.val_list))


def scale(image, bits):
    img = image.astype(np.float32) / (2**bits - 1)  # only support uint8, uint16 now.
    return img


def refer_based_normalize(image: np.array, refer: np.array):
    """a wrapper of monai.transforms.ReferenceBasedNormalizeIntensity

    Args:
        image (np.array): input image
        refer (np.array): refer image
    """
    img = image.astype(np.float32)
    img = torch.from_numpy(img)
    refer = refer.astype(np.float32)
    refer = torch.from_numpy(refer)
    normalize_intensity = monai.transforms.ReferenceBasedNormalizeIntensityd()
    img_normalized = normalize_intensity({"image": img, "target": refer})["image"]
    return img_normalized.numpy()


def normalize(image: np.array):
    img = image.astype(np.float32)
    img = torch.from_numpy(img)
    normalize_intensity = monai.transforms.NormalizeIntensity()
    img_normalized = normalize_intensity(img)
    return img_normalized.numpy()


def compare_images(path1, path2, dim=2, pred=False):
    # Load the two images
    if dim == 2:
        image1 = AICSImage(path1).get_image_data("YX")
        image2 = AICSImage(path2).get_image_data("YX")
    elif dim == 3:
        image1 = AICSImage(path1).get_image_data("ZYX")
        image2 = AICSImage(path2).get_image_data("ZYX")
    else:
        raise NotImplementedError(f"dim {dim} is not supported now!")
    bits = np.dtype(image1.dtype).itemsize * 8
    image1 = normalize(image1)
    image2 = normalize(image2)

    # Calculate metrics
    mse_value = mse(image1, image2)
    ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())
    psnr_value = psnr(image1, image2, data_range=image1.max() - image1.min())
    corr_value = np.corrcoef(image1.ravel(), image2.ravel())[0, 1]
    if not pred:
        storage_ratio = compression_ratio_space_savings(
            str(path1).replace("_decoded.tiff", "_encoded"),
            np.array(image1.shape).prod(),
            bits,
        )
    else:
        storage_ratio = 1
    return mse_value, ssim_value, psnr_value, corr_value, storage_ratio


def compression_ratio_space_savings(image_path, values=924 * 624, bits=16):
    ratio = (os.stat(image_path).st_size * 8) / (values * bits)
    return ratio


# input= dir structure: types and raw data
def plot_data(random_image, compression_techniques: Optional[Sequence] = ["LERC"]):
    root_dir = random_image.parent
    print(f"We choose {random_image.name} as the illustration.")
    for dir in [
        i
        for i in root_dir.glob("*")
        if (
            (i / random_image.name).exists()
            or (i / str(random_image.stem + "_decoded.tiff")).exists()
        )
        and i.name in compression_techniques
    ]:
        try:
            print(dir.name)
            # Find all immediate subdirectories
            subdirectories = [subdir for subdir in dir.iterdir() if subdir.is_dir()]
            spacing = 0.05

            # Calculate the width and height of a subplot based on the space available and desired spacing
            width = (1 - 4 * spacing) / 3
            height = (1 - 3 * spacing) / 2

            # To center-align the subplots in the second row,
            # we'll adjust their x-positions based on the spacing and width.
            # Calculate the total space available for the second row
            total_space = 1 - 2 * spacing
            space_occupied_by_subplots = 2 * width
            space_left = total_space - space_occupied_by_subplots

            # Calculate the new starting x-position for the fourth subplot to center-align them
            new_start_x = spacing + space_left / 2.3

            # Adjust positions for the first row
            pos1 = [new_start_x, 0.5 + spacing, width, height]
            pos2 = [pos1[0] + width + spacing, pos1[1], width, height]
            pos3 = [spacing, spacing, width, height]
            # Adjust positions for the center-aligned subplots in the second row
            pos4 = [pos3[0] + width + spacing, spacing, width, height]
            pos5 = [pos4[0] + width + spacing, spacing, width, height]

            input_img = AICSImage(random_image).get_image_data("YX")

            if dir.name in ["LERC", "JPEGXR", "JPEG_2000_LOSSY"]:
                compression_img = AICSImage(dir / random_image.name).get_image_data(
                    "YX"
                )
            else:
                compression_img = AICSImage(
                    dir / str(random_image.stem + "_decoded.tiff")
                ).get_image_data("YX")

            # Plot
            plt.figure(figsize=(12, 6))
            plt.clf()
            ax1 = plt.axes(pos1)  # 第一行第一个位置
            plt.imshow(input_img, cmap="gray")
            plt.title(f"Input")
            scalebar1 = ScaleBar(0.108333, "um", frameon=False, color="white")
            ax1.add_artist(scalebar1)
            plt.axis("off")

            ax2 = plt.axes(pos2)
            plt.imshow(compression_img, cmap="gray")
            plt.title("Compression")
            scalebar2 = ScaleBar(0.108333, "um", frameon=False, color="white")
            ax2.add_artist(scalebar2)
            plt.axis("off")

            if len(subdirectories):
                if dir.name in ["LERC", "JPEGXR", "JPEG_2000_LOSSY"]:
                    prediction_img = AICSImage(
                        dir / "prediction" / Path(random_image.stem + "_pred.tiff")
                    ).get_image_data("YX")
                else:
                    prediction_img = AICSImage(
                        dir
                        / "prediction"
                        / Path(random_image.stem + "_decoded_pred.tiff")
                    ).get_image_data("YX")
                groundtruth_img = AICSImage(
                    root_dir / random_image.name.replace("IM", "GT")
                ).get_image_data("YX")
                original_prediction = AICSImage(
                    dir.parent
                    / "original"
                    / "prediction"
                    / Path(random_image.stem + "_pred.tiff")
                ).get_image_data("YX")

                ax3 = plt.axes(pos3)
                plt.imshow(prediction_img, cmap="gray")
                plt.title("Compressed Prediction")
                scalebar3 = ScaleBar(0.108333, "um", frameon=False, color="white")
                ax3.add_artist(scalebar3)
                plt.axis("off")

                ax4 = plt.axes(pos4)
                gt = NormalizeIntensity()(original_prediction)
                plt.imshow(gt, cmap="gray")
                plt.title("Original Prediction")
                scalebar4 = ScaleBar(0.108333, "um", frameon=False, color="white")
                ax4.add_artist(scalebar4)
                plt.axis("off")

                ax5 = plt.axes(pos5)
                gt = NormalizeIntensity()(groundtruth_img)
                plt.imshow(gt, cmap="gray")
                plt.title("Ground Truth (Normalized Intensity)")
                scalebar5 = ScaleBar(0.108333, "um", frameon=False, color="white")
                ax5.add_artist(scalebar5)
                plt.axis("off")
            plt.show()
        except FileNotFoundError as e:
            print(e)
            continue


def write_metrics(
    holdout_path_2d, output_path, pred=False, compare_with_original=False
):
    table_data = []
    for dir in [i for i in holdout_path_2d.iterdir() if i.is_dir()]:
        mses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        corrs = AverageMeter()
        storages = AverageMeter()

        if not pred:
            for image in Path(dir).glob("*.tiff"):
                (
                    mse_value,
                    ssim_value,
                    psnr_value,
                    corr_value,
                    storage_ratio,
                ) = compare_images(
                    image,
                    dir.parent / "original" / image.name.replace("_decoded", ""),
                    pred=pred,
                )
        elif not compare_with_original:
            for image in (Path(dir) / "prediction").glob("*.tiff"):
                (
                    mse_value,
                    ssim_value,
                    psnr_value,
                    corr_value,
                    storage_ratio,
                ) = compare_images(
                    image,
                    dir.parent
                    / image.name.replace("_IM", "")
                    .replace("_decoded", "")
                    .replace("_pred", "_GT"),
                    pred=pred,
                )
        else:
            for image in (Path(dir) / "prediction").glob("*.tiff"):
                (
                    mse_value,
                    ssim_value,
                    psnr_value,
                    corr_value,
                    storage_ratio,
                ) = compare_images(
                    image,
                    dir.parent
                    / "original"
                    / "prediction"
                    / image.name.replace("_decoded", ""),
                    pred=pred,
                )

        mses.update(mse_value)
        ssims.update(ssim_value)
        psnrs.update(psnr_value)
        corrs.update(corr_value)
        storages.update(storage_ratio)

        table_data.append(
            {
                "dirname": dir.name.replace("_", "-"),
                "mse": mses.avg,
                "mse_std": mses.std,
                "ssim": ssims.avg,
                "ssim_std": ssims.std,
                "psnr": psnrs.avg,
                "psnr_std": psnrs.std,
                "corr": corrs.avg,
                "corr_std": corrs.std,
                "storage": storages.avg,
                "storage_std": storages.std,
            }
        )
    df = pd.DataFrame(table_data)
    df = df.round(4)
    df.to_csv(Path(output_path), index=False, sep=",", decimal=".")
    print(df.head(len(table_data)))
