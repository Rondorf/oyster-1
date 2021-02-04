import numpy as np
import torch


@torch.no_grad()
def augment_obs(obs, random=True):
    obs = center_crop_images(obs, 64)
    if random:
        obs = random_translate(obs, 70)
    else:
        obs = center_translate(obs, 70)
    return obs


def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    with torch.no_grad():
        n, c, h, w = imgs.shape
        assert size >= h and size >= w
        outs = torch.zeros((n, c, size, size), device=imgs.device, dtype=imgs.dtype)
        h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
        w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
        for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
            out[:, h1:h1 + h, w1:w1 + w] = img
        if return_random_idxs:  # So can do the same to another set of imgs.
            return outs, dict(h1s=h1s, w1s=w1s)
        return outs


def center_translate(image, size):
    with torch.no_grad():
        n, c, h, w = image.shape
        assert size >= h and size >= w
        outs = torch.zeros((n, c, size, size), device=image.device, dtype=image.dtype).cuda()
        h1s = np.zeros(n, dtype=np.int) + ((size - h) // 2)
        w1s = np.zeros(n, dtype=np.int) + ((size - w) // 2)
        for out, img, h1, w1 in zip(outs, image, h1s, w1s):
            out[:, h1:h1 + h, w1:w1 + w] = img
        return outs


def center_crop_image(image, output_size):
    with torch.no_grad():
        h, w = image.shape[1:]
        new_h, new_w = output_size, output_size

        top = (h - new_h)//2
        left = (w - new_w)//2

        image = image[:, top:top + new_h, left:left + new_w]
        return image


def center_crop_images(image, output_size):
    with torch.no_grad():
        h, w = image.shape[2:]
        new_h, new_w = output_size, output_size

        top = (h - new_h)//2
        left = (w - new_w)//2

        image = image[:, :, top:top + new_h, left:left + new_w]
        return image

