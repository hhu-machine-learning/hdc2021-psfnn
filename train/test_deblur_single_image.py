from PIL import Image
import numpy as np

def process_tiled(image, size, padding, process_tile):
    h, w = image.shape
    reassembled = np.zeros((h, w))

    w_new = (w + size - 1) // size * size
    h_new = (h + size - 1) // size * size

    y_missing = h_new - h
    x_missing = w_new - w

    image = np.pad(image, [(padding, padding + y_missing), (padding, padding + x_missing)], mode='reflect')

    for y in range(0, h_new, size):
        for x in range(0, w_new, size):
            print("processing tile", x, y, x + size, y + size, "of", w, h)
            tile = image[y:y+size+2*padding, x:x+size+2*padding]

            target = reassembled[y:y+size, x:x+size]

            # must crop tile to tile[padding:-padding, padding:-padding]
            tile = process_tile(tile)

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

def main():
    step = 9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 0:
        # Basic UNet
        from net import UNet
        net = UNet(3, 1).to(args["device"])

    if 0:
        # Neural net from IndexNet matting
        from hlmobilenetv2 import hlmobilenetv2
        net = hlmobilenetv2(
            pretrained=True,
            freeze_bn=True,
            output_stride=32,
            input_size=320,
            apply_aspp=False,
            conv_operator="std_conv",
            #decoder="indexnet",
            decoder="indexnet",
            decoder_kernel_size=5,
            indexnet="depthwise",
            index_mode="m2o",
            use_nonlinear=False,
            use_context=False,
            sync_bn=False,
        ).to(args["device"])

    if 1:
        # Neural net from FBA matting
        from networks.models import fba_decoder, MattingModule, ResnetDilated
        from networks.resnet_GN_WS import l_resnet50
        orig_resnet = l_resnet50()
        net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        net_decoder = fba_decoder(num_output_channels=1, batch_norm=False)
        net = MattingModule(net_encoder, net_decoder).to(device)
        path = "../output/blur_exp028_train_blur_from_model_5000"
        path = "../output/blur_exp029_train_blur_from_images_34000"
        path = "../output/blur_natural_plus_hdc_exp029_train_blur_from_images_19000"
        path = "../output/blur_natural_plus_hdc_exp029_train_blur_from_images_50000"
        path = "../output/blur_exp030_1000.pth"
        path = f"../output/blur_exp030_step_{step}_batch_50000.pth"
        net.load_state_dict(torch.load(path))
        net.eval()

    padding = 160
    tile_size = 640
    # actual tile size additionally includes 2 * padding

    def process_tile(tile):
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

    for font in ["Times", "Verdana"]:
        for sample in [1, 100]:
            path = f"deblurred_{font}_sample_{sample}_step_{step}.png"
            print("processing", path)

            blurry = load(step=step, cam=2, sample=sample, font=font)

            #blurry = cv2.resize(blurry, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            deblurred = process_tiled(blurry, tile_size, padding, process_tile)

            #2360 x 1460

            #factor = 320

            h, w = blurry.shape

            """
            blurry = np.pad(blurry, [
                (0, (h + factor - 1) // factor * factor),
                (0, (w + factor - 1) // factor * factor),
            ], mode='reflect')
            """

            #blurry = cv2.resize(blurry, (320 * 4, 320 * 4), interpolation=cv2.INTER_AREA)

            #deblurred = process_tile(blurry)

            #deblurred = cv2.resize(deblurred, (w, h), interpolation=cv2.INTER_AREA)

            Image.fromarray(np.clip(deblurred * 255, 0, 255).astype(np.uint8)).save(path)

    """
    show([
        ("ground truth", sharp),
        ("input", blurry),
        ("deblurred", deblurred),
    ], 3, 1)
    """

if __name__ == "__main__":
    main()
