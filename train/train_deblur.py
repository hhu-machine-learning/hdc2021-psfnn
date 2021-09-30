from PIL import Image
import numpy as np
import os, glob, time, json, random, shutil
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
import div2k
from dewarp import warp
from load_hdc import load_hdc
#sys.path.append("../exp028_train_blur_from_model")
#sys.path.append("../exp028_train_blur_from_model/networks")

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

def load_image_pairs_div2k(n, blur_model, train):
    psf_radius = blur_model.psf_radius

    pairs = []
    for i, sharp in enumerate(div2k.Dataset(train=train)):
        print(f"load div2k {i:3d}")
        sharp = sharp[:, :, 1]

        sharp = sharp.astype(np.float32) / 255.0

        # blur the sharp image with estimated PSF, then crop sharp image
        # because blur also crops and training needs same sizes
        # TODO try reflect boundary condition for blur
        with torch.no_grad():
            sharp = torch.tensor(sharp, device=blur_model.psf.device)

            blurry = blur_model(sharp)

            sharp = sharp[psf_radius:-psf_radius, psf_radius:-psf_radius]

            sharp = np.clip(sharp.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            blurry = np.clip(blurry.cpu().numpy() * 255, 0, 255).astype(np.uint8)

        #path = f"../output/div2k_step4_blurred_train_{train}/{i:04d}.bmp"
        #print("loading", path)

        pairs.append((sharp, blurry))

        if i >= n: break

    return pairs
"""
def load(step, cam, sample, font):
    fix_for_inconsistent_naming_scheme = {
        "Times": "timesR",
        "Verdana": "verdanaRef",
    }[font]

    path = f"~/data/hdc2021/step{step}/{font}/CAM{cam:02d}/focusStep_{step}_{fix_for_inconsistent_naming_scheme}_size_30_sample_{sample:04d}.tif"
    path = os.path.expanduser(path)

    print("load", path)

    image = Image.open(path)

    image = np.float32(image) / 65535.0

    return image

def load_image_pairs_hdc(step, font, samples):
    psf_radius = 100

    pairs = []
    for sample in samples:
        sharp = load(step=step, cam=1, sample=sample, font=font)

        # TODO check if float instead of uint8 could fit into memory
        sharp = np.clip(sharp * 255, 0, 255).astype(np.uint8)

        sharp = sharp[psf_radius:-psf_radius, psf_radius:-psf_radius]

        path = f"../output/{font}_step{step}_blurred/{sample:04d}.bmp"

        print("load", path)

        blurry = np.array(Image.open(path))

        pairs.append((sharp, blurry))

    return pairs

def load_image_pairs_hdc2(step, font, train):
    psf_radius = 100

    pairs = []
    for sample in range(1, 11):
        sharp = load(step=step, cam=1, sample=sample, font=font)

        # TODO check if float instead of uint8 could fit into memory
        sharp = np.clip(sharp * 255, 0, 255).astype(np.uint8)

        if train:
            sharp = sharp[psf_radius:-psf_radius, psf_radius:-psf_radius]

            path = f"../output/{font}_step{step}_blurred/{sample:04d}.bmp"

            blurry = np.array(Image.open(path))
        else:
            blurry = load(step=step, cam=2, sample=sample, font=font)

            # TODO check if float instead of uint8 could fit into memory
            blurry = np.clip(blurry * 255, 0, 255).astype(np.uint8)

        pairs.append((sharp, blurry))

    return pairs
"""
def load_image_pairs_hdc_original(step, fonts, samples, blur_model):
    M = blur_model.M

    pairs = []
    for sample in samples:
        for font in fonts:
            print(f"load hdc sample {sample} font {font}")
            sharp  = load_hdc(step=step, cam=1, sample=sample, font=font)
            blurry = load_hdc(step=step, cam=2, sample=sample, font=font)

            # dewarp blurry image to make NN job easier
            with torch.no_grad():
                blurry = torch.tensor(blurry, device=M.device)
                blurry = warp(blurry, M)
                blurry = blurry.cpu().numpy()

            # TODO check if float instead of uint8 could fit into memory
            sharp = np.clip(sharp * 255, 0, 255).astype(np.uint8)
            blurry = np.clip(blurry * 255, 0, 255).astype(np.uint8)

            pairs.append((sharp, blurry))

    return pairs

def make_batch(batch_size, crop_size, device, augmentation_noise, samples, **kwargs):
    sharps = torch.zeros((batch_size, 1, crop_size, crop_size))
    blurrs = torch.zeros((batch_size, 3, crop_size, crop_size))

    for i in range(batch_size):
        sharp, blurry = random.choice(samples)

        h, w = sharp.shape

        x = np.random.randint(w - crop_size)
        y = np.random.randint(h - crop_size)

        sharps[i, 0] = torch.tensor(sharp [y:y+crop_size, x:x+crop_size].astype(np.float32) / 255.0)

        blurry_crop = torch.tensor(blurry[y:y+crop_size, x:x+crop_size].astype(np.float32) / 255.0)

        blurrs[i, 0] = blurry_crop
        blurrs[i, 1] = blurry_crop
        blurrs[i, 2] = blurry_crop
        #blurrs[i, 3] = blurry_crop

    sharps = sharps.to(device)
    blurrs = blurrs.to(device)

    with torch.no_grad():
        blurrs += augmentation_noise * torch.randn(sharps.shape, device=device)

    return blurrs, sharps

def train(net, args, num_batches, lr, device, train_samples, test_samples, step, **kwargs):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    test_args = update(update(args, "batch_size", 10), "augmentation_noise", 0.0)

    test_inputs, test_targets = make_batch(samples=test_samples, **test_args)

    test_targets = test_targets.to(device)
    test_inputs = test_inputs.to(device)

    #scheduler = MultiStepLR(optimizer, milestones=[3, 6], gamma=0.1)

    shutil.rmtree("preview", ignore_errors=True)
    os.makedirs("preview")

    log = []
    for batch in range(1, 1 + num_batches):
        net.train()
        t0 = time.perf_counter()

        train_inputs, train_targets = make_batch(samples=train_samples, **args)

        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        train_outputs = net(train_inputs)

        optimizer.zero_grad()

        #train_loss = F.mse_loss(train_outputs, train_targets)
        train_mse = torch.mean(torch.square(train_outputs - train_targets))

        train_mse.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Print first epoch, first batch of every epoch and last batch
        if batch <= 10 or batch % 10 == 0:
            with torch.no_grad():
                net.eval()

                test_outputs = net(test_inputs)

                test_mse = torch.mean(torch.square(test_outputs - test_targets))

            log.append({
                "batch": int(batch),
                "test_mse": float(test_mse.item()),
                "train_mse": float(train_mse.item()),
                "make_batch": float(t1 - t0),
                "train_batch": float(t2 - t1),
            })

            print(" - ".join(f"{key} {value:10.8f}" if isinstance(value, float) else f"{key} {value:10d}" for key, value in log[-1].items()))
            with open("log.json", "w") as f: json.dump(log, f, indent=4)

        if batch == 1 or batch % 100 == 0:
            show([
                ("train_input", train_inputs[0, :1]),
                ("train_target", train_targets[0]),
                ("train_output", train_outputs[0]),
                ("test_input", test_inputs[0, :1]),
                ("test_target", test_targets[0]),
                ("test_output", test_outputs[0]),
            ], path=f"preview/batch_{batch}.png", nx=3, ny=2)

        if batch == 1 or batch % 1000 == 0:
            parent_dir = os.path.split(os.path.abspath("."))[1]
            os.makedirs("../tmp", exist_ok=True)
            path = f"../tmp/blur_{parent_dir}_step_{step}_batch_{batch}.pth"
            torch.save(net.state_dict(), path)
            print("saved net to", path)


        #scheduler.step()

    return optimizer

def main():
    step = 10
    #font = "Verdana"
    fonts = ["Times", "Verdana"]

    blur_model = torch.load(f"psfs/step_{step}.pth")

    args = {
        "step": step,
        "crop_size": 320,
        "batch_size": 2,
        "num_batches": 50000,
        "lr": 1e-4,
        #"train_samples": np.arange(1, 81),
        #"test_samples": np.arange(81, 101),
        "augmentation_noise": 0.03,
        #"fonts": ["Times", "Verdana"],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        #"train_samples": load_image_pairs_div2k(n=500, train=True),
        #"test_samples": load_image_pairs_div2k(n=100, train=False),
        #"train_samples": load_image_pairs_hdc(step=step, font=font, samples=list(range(1, 91))),
        #"test_samples": load_image_pairs_hdc(step=step, font=font, samples=list(range(91, 101))),
        #"train_samples": load_image_pairs_hdc2(step=step, font=font, train=True),
        #"test_samples": load_image_pairs_hdc2(step=step, font=font, train=False),

        #"train_samples": load_image_pairs_hdc_original(step=step, fonts=fonts, samples=list(range(1, 91))) + load_image_pairs_div2k(n=500, train=True),
        #"test_samples": load_image_pairs_hdc_original(step=step, fonts=fonts, samples=list(range(91, 101))) + load_image_pairs_div2k(n=100, train=False),

        "train_samples": load_image_pairs_hdc_original(step=step, fonts=fonts, blur_model=blur_model, samples=list(range(1, 91))) + load_image_pairs_div2k(n=500, blur_model=blur_model, train=True),
        "test_samples": load_image_pairs_hdc_original(step=step, fonts=fonts, blur_model=blur_model, samples=list(range(91, 101))) + load_image_pairs_div2k(n=100, blur_model=blur_model, train=False),

        # for quick testing
        #"train_samples": load_image_pairs_hdc_original(step=step, fonts=fonts, blur_model=blur_model, samples=list(range(1, 2))) + load_image_pairs_div2k(n=1, blur_model=blur_model, train=True),
        #"test_samples": load_image_pairs_hdc_original(step=step, fonts=fonts, blur_model=blur_model, samples=list(range(100, 101))) + load_image_pairs_div2k(n=1, blur_model=blur_model, train=False),
    }

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
        net = MattingModule(net_encoder, net_decoder).to(args["device"])

    train(net, args, **args)

if __name__ == "__main__":
    main()
