from PIL import Image
import numpy as np

def process_tiled(image, size, padding, process_tile):
    h, w = image.shape
    reassembled = np.zeros((h, w), dtype=image.dtype)

    w_new = (w + size - 1) // size * size
    h_new = (h + size - 1) // size * size

    y_missing = h_new - h
    x_missing = w_new - w

    image = np.pad(image, [(padding, padding + y_missing), (padding, padding + x_missing)], mode='reflect')

    for y in range(0, h_new, size):
        for x in range(0, w_new, size):
            print(f"processing tile {x:04d}, {y:04d}, {x + size:04d}, {y + size:04d} of {w}-by-{h} image")
            tile = image[y:y+size+2*padding, x:x+size+2*padding]

            target = reassembled[y:y+size, x:x+size]

            # must crop tile to tile[padding:-padding, padding:-padding]
            tile = process_tile(tile)

            assert(tile.shape == (size, size))

            # crop tile if tile is too large
            tile = tile[:target.shape[0], :target.shape[1]]

            target[...] = tile

    return reassembled

def process_tiles(image, size, padding, process_tiles):
    h, w = image.shape
    reassembled = np.zeros((h, w), dtype=image.dtype)

    w_new = (w + size - 1) // size * size
    h_new = (h + size - 1) // size * size

    y_missing = h_new - h
    x_missing = w_new - w

    image = np.pad(image, [(padding, padding + y_missing), (padding, padding + x_missing)], mode='reflect')

    tiles = []
    for y in range(0, h_new, size):
        for x in range(0, w_new, size):
            tile = image[y:y+size+2*padding, x:x+size+2*padding]
            tiles.append(tile)

    tiles = process_tiles(tiles)

    k = 0
    for y in range(0, h_new, size):
        for x in range(0, w_new, size):
            target = reassembled[y:y+size, x:x+size]

            # must crop tile to tile[padding:-padding, padding:-padding]
            tile = tiles[k]
            k += 1

            assert(tile.shape == (size, size))

            # crop tile if tile is too large
            tile = tile[:target.shape[0], :target.shape[1]]

            target[...] = tile

    return reassembled

if 0:
    image = np.array(Image.open("../lemur.png").convert("L"))

    #padded = reflection_pad(image, 50, 100, 200, 300)
    #import matplotlib.pyplot as plt
    #plt.imshow(padded, cmap='gray')
    #plt.show()
    #exit(0)

    size = 73
    padding = 31

    def crop(tile):
        return tile[padding:-padding, padding:-padding]

    reassembled = process_tiled(image, size, padding, crop)

    assert(np.allclose(image, reassembled))

    import matplotlib.pyplot as plt
    plt.imshow(reassembled, cmap='gray')
    plt.show()

from PIL import Image
import numpy as np
import os, glob, time, json, random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
import div2k
from dewarp import warp
#sys.path.append("../exp028_train_blur_from_model")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def update(d, key, value):
    d = dict(d)
    d[key] = value
    return d

def show(images, nx, ny, path=None):
    plt.clf()
    plt.close()
    for i, image in enumerate(images):
        plt.subplot(ny, nx, 1 + i)

        if len(image) == 2:
            title, image = image
            plt.title(title)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image.transpose(1, 2, 0).squeeze()

        plt.imshow(np.clip(image, 0, 1), cmap='gray', vmin=0, vmax=1)

    plt.tight_layout()

    if path:
        plt.savefig(path)
    else:
        plt.show()

def load_image_pairs(n, train):
    psf_radius = 100

    pairs = []
    for i, sharp in enumerate(div2k.Dataset(train=train)):
        sharp = sharp[:, :, 1]

        sharp = sharp[psf_radius:-psf_radius, psf_radius:-psf_radius]

        path = f"../output/div2k_step4_blurred_train_{train}/{i:04d}.bmp"
        print("loading", path)

        blurry = np.array(Image.open(path))

        pairs.append((sharp, blurry))

        if i >= n: break

    return pairs

def load(step, cam, sample, font):
    fix_for_inconsistent_naming_scheme = {
        "Times": "timesR",
        "Verdana": "verdanaRef",
    }[font]

    path = f"~/data/hdc2021/step{step}/{font}/CAM{cam:02d}/focusStep_{step}_{fix_for_inconsistent_naming_scheme}_size_30_sample_{sample:04d}.tif"
    path = os.path.expanduser(path)

    image = Image.open(path)

    image = np.float32(image) / 65535.0

    return image
import sys
import util

# Example:
# python3 exp024_baseline_just_copy_images.py ~/data/hdc2021/step5/Times/CAM02/ output/exp024/step5/Times/ 5

import numpy as np
import cv2

# Neural net from FBA matting

net = None
device = torch.device("cuda")

def init(step):
    global net
    from networks.models import fba_decoder, MattingModule, ResnetDilated
    from networks.resnet_GN_WS import l_resnet50
    orig_resnet = l_resnet50()
    net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
    net_decoder = fba_decoder(num_output_channels=1, batch_norm=False)
    net = MattingModule(net_encoder, net_decoder).to(device)
    path = "../output/blur_exp030_1000.pth"
    path = f"../output/blur_exp030_step_{step}_batch_50000.pth"
    net.load_state_dict(torch.load(path))
    net.eval()

def deblur_image(blurry, step):
    if net is None:
        init(step)

    # Apply sophisticated image deblurring algorithm

    blur_model = torch.load(f"psfs/step_{step}.pth")

    M = blur_model.M

    with torch.no_grad():
        blurry = torch.tensor(blurry, device=M.device)
        blurry = warp(blurry, M)
        blurry = blurry.cpu().numpy()

    padding = 160
    tile_size = 640
    # actual tile size additionally includes 2 * padding

    def process_tile_batch(tiles):
        batch_size = 16

        result = []

        for i in range(0, len(tiles), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(tiles))):
                tmp = tiles[j]
                tmp = tmp[None, None, :, :]
                tmp = np.concatenate([tmp] * 3, axis=1)
                batch.append(tmp)
            batch = np.concatenate(batch, axis=0)

            batch = torch.tensor(batch, device=device)
            batch = net(batch)

            for tmp in batch:
                tmp = tmp[0, :, :]
                tmp = tmp.detach().cpu().numpy()
                tmp = tmp[padding:-padding, padding:-padding]
                result.append(tmp)

        return result

    def process_tile_single(tile):
        with torch.no_grad():
            tmp = tile
            tmp = tmp[None, None, :, :]
            tmp = np.concatenate([tmp] * 3, axis=1)
            tmp = torch.tensor(tmp, device=device)
            #tmp += 0.03 * torch.randn(tmp.shape, device=tmp.device)
            tmp = net(tmp)
            tmp = tmp[0, 0, :, :]
            tmp = tmp.cpu().numpy()
            tmp = tmp[padding:-padding, padding:-padding]
        return tmp

    t = time.perf_counter()

    #deblurred_image = process_tiles(blurry, tile_size, padding, process_tile_batch)
    deblurred_image = process_tiled(blurry, tile_size, padding, process_tile_single)

    dt = time.perf_counter() - t
    print(dt, "seconds")

    return deblurred_image

def main():
    if len(sys.argv) != 4:
        if len(sys.argv) == 2:
            _, step = sys.argv
        elif len(sys.argv) == 1:
            step = 9
        else:
            print("Usage: python3 filename.py <directory of input image files> <directory of output image files> <step (integer from 0 to 19)>")
            return

        font = "Verdana"
        font = "Times"

        import os
        input_dir = os.path.expanduser(f"~/data/hdc2021/step{step}/{font}/CAM02/")
        output_dir = f"../output/blur_exp030_1000/step{step}/{font}/"
    else:
        _, input_dir, output_dir, step = sys.argv

    util.process_images(deblur_image, input_dir, output_dir, step, parallel=False)

if __name__ == "__main__":
    main()
