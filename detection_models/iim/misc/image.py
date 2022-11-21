__all__ = ['get_points_on_image']

import os

import PIL
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as standard_transforms

import detection_models.iim.misc.transforms as own_transforms


GPU_ID = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

torch.backends.cudnn.benchmark = True
netName = 'HR_Net'  # options: HR_Net,VGG16_FPN
default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_boxInfo_from_Binar_map(binar_numpy, min_area=3):
    binar_numpy = binar_numpy.squeeze().astype(np.uint8)
    assert binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binar_numpy, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    return points


def get_points_on_image(img_raw, model, device=None):
    if isinstance(img_raw, PIL.Image.Image):
        return get_points_on_pil_image(img_raw, model, device)
    elif isinstance(img_raw, np.ndarray):
        return get_points_on_np_image(img_raw, model, device)
    else:
        raise ValueError(str(type(img_raw)), " is not supported")


def get_points_on_pil_image(img: PIL.Image.Image, net, device) -> np.array:
    if img.mode == 'L':
        img = img.convert('RGB')
    img = img_transform(img)[None, :, :, :]
    points = get_points_on_tensor(img, net, device)
    return points


def get_points_on_np_image(img: np.ndarray,  net, device) -> np.array:
    img = img_transform(img)[None, :, :, :]
    points = get_points_on_tensor(img, net, device)
    return points

    points = get_points_on_tensor(img_raw, model, device)
    return points


def get_points_on_tensor(img: torch.Tensor, net, device=None):
    if not device:
        device = default_device

    slice_h, slice_w = 512, 1024

    with torch.no_grad():
        img = Variable(img).to(device)
        b, c, h, w = img.shape
        crop_imgs, crop_dots, crop_masks = [], [], []
        if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
            [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
        else:
            if h % 16 != 0:
                pad_dims = (0, 0, 0, 16 - h % 16)
                h = (h // 16 + 1) * 16
                img = F.pad(img, pad_dims, "constant")

            if w % 16 != 0:
                pad_dims = (0, 16 - w % 16, 0, 0)
                w = (w // 16 + 1) * 16
                img = F.pad(img, pad_dims, "constant")

            for i in range(0, h, slice_h):
                h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                for j in range(0, w, slice_w):
                    w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                    crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                    mask = torch.zeros(1, 1, img.size(2), img.size(3)).cpu()
                    mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks = torch.cat(crop_imgs, dim=0), torch.cat(crop_masks, dim=0)

            # forward may need repeatng
            crop_preds, crop_thresholds = [], []
            nz, period = crop_imgs.size(0), 4
            for i in range(0, nz, period):
                [crop_threshold, crop_pred, __] = [i.cpu() for i in
                                                   net(crop_imgs[i:min(nz, i + period)], mask_gt=None, mode='val')]
                crop_preds.append(crop_pred)
                crop_thresholds.append(crop_threshold)

            crop_preds = torch.cat(crop_preds, dim=0)
            crop_thresholds = torch.cat(crop_thresholds, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros(b, 1, h, w).cpu()
            pred_threshold = torch.zeros(b, 1, h, w).cpu().float()
            for i in range(0, h, slice_h):
                h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                for j in range(0, w, slice_w):
                    w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                    pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                    pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                    idx += 1
            mask = crop_masks.sum(dim=0)
            pred_map = (pred_map / mask)
            pred_threshold = (pred_threshold / mask)

        a = torch.ones_like(pred_map)
        b = torch.zeros_like(pred_map)
        binar_map = torch.where(pred_map >= pred_threshold, a, b)

        points = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())

        return points