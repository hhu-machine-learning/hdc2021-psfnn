#!/usr/bin/env python3
import os, sys, time
sys.path.append("train")
from dewarp import warp
from networks.models import fba_decoder, MattingModule, ResnetDilated
from networks.resnet_GN_WS import l_resnet50
from PIL import Image
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_FILE_EXTENSIONS = ["png", "tiff", "tif", "gif", "jpg", "jpeg", "bmp"]

def split_ext(filename):
    return filename.rsplit(".", maxsplit=1)

def load_image(path):
    # Load an image in range [0, 1] as a float32 numpy array from a path
    image = np.array(Image.open(path))

    if image.dtype == np.uint8:
        return np.float32(image) / 255.0
    elif image.dtype == np.uint16:
        return np.float32(image) / 65535.0
    else:
        raise ValueError(f"Unsupported image dtype {image.dtype} for {path}")

def save_image(path, image):
    assert(isinstance(image, np.ndarray))
    assert(image.dtype == np.float32)

    image_uint8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

    Image.fromarray(image_uint8).save(path)

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
            print(f"processing tile {x:4d}, {y:4d}, {x + size:4d}, {y + size:4d} of {w}-by-{h} image")
            tile = image[y:y+size+2*padding, x:x+size+2*padding]

            target = reassembled[y:y+size, x:x+size]

            # must crop tile to tile[padding:-padding, padding:-padding]
            tile = process_tile(tile)

            assert(tile.shape == (size, size))

            # crop tile if tile is too large
            tile = tile[:target.shape[0], :target.shape[1]]

            target[...] = tile

    return reassembled

def deblur(blurry, net, step, device):
    assert(isinstance(blurry, np.ndarray))
    assert(blurry.dtype == np.float32)

    if step in ["0", "1", "2", "3"]:
        print("Not doing any deblurring for steps before 4 because that would be a waste of energy since it passes the OCR tests anyway")
        return blurry

    blur_model = torch.load(f"train/psfs/step_{step}.pth")

    M = blur_model.M

    with torch.no_grad():
        blurry = torch.tensor(blurry, device=M.device)
        blurry = warp(blurry, M)
        blurry = blurry.cpu().numpy()

    # Actual tile size additionally includes 2 * padding
    tile_size = 640
    padding = 160

    def process_tile(tile):
        with torch.no_grad():
            tmp = tile
            tmp = tmp[None, None, :, :]
            tmp = np.concatenate([tmp] * 3, axis=1)
            tmp = torch.tensor(tmp, device=device)
            tmp = net(tmp)
            tmp = tmp[0, 0, :, :]
            tmp = tmp.cpu().numpy()
            tmp = tmp[padding:-padding, padding:-padding]
        return tmp

    t = time.perf_counter()

    deblurred_image = process_tiled(blurry, tile_size, padding, process_tile)

    dt = time.perf_counter() - t
    print(dt, "seconds")

    return deblurred_image

def main():
    step = "3"

    if len(sys.argv) == 4:
        _, input_dir, output_dir, step = sys.argv
    else:
        print("Usage:\n    python3 main.py path/to/input/files path/to/output/files 3\n")

        # TODO
        font = "Times"
        input_dir = os.path.expanduser(f"~/data/hdc2021/step{step}/{font}/CAM02/")
        output_dir = f"tmp/step{step}/{font}"
        print("Choosing default input dir", input_dir)
        print("Choosing default output dir", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")

    if step in ["0", "1", "2", "3"]:
        net = None
    else:
        orig_resnet = l_resnet50()
        net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        net_decoder = fba_decoder(num_output_channels=1, batch_norm=False)
        net = MattingModule(net_encoder, net_decoder).to(device)

        path = f"tmp/blur_exp030_step_{step}_batch_50000.pth"

        if not os.path.isfile(path):
            print(f"Now downloading neural network for step {step} (132 MB).")
            print("This might take a while...")
            url = f"https://asdf10.com/blur_exp030_step_{step}_batch_50000.pth"
            urllib.request.urlretrieve(url, path)
            print("Network has been downloaded.")

        # TODO map to device on load
        net.load_state_dict(torch.load(path))
        net.eval()

    for input_filename in sorted(os.listdir(input_dir)):
        name, ext = split_ext(input_filename)

        # Skip files where file extension does not look like image files
        if ext.lower() not in IMAGE_FILE_EXTENSIONS: continue

        input_path = os.path.join(input_dir, input_filename)
        output_path = os.path.join(output_dir, name + ".png")

        print("loading", input_path)

        blurry_image = load_image(input_path)

        deblurred_image = deblur(blurry_image, net, step, device)

        save_image(output_path, deblurred_image)

        print("saved to", output_path)
        print()

if __name__ == "__main__":
    main()
