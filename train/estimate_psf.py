import matplotlib.pyplot as plt
import os
import collections
import time
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from blur_model import BlurModel
from dewarp import get_dewarping_matrix
from load_hdc import load_hdc

def optimize_model(model, sharp, blurry):
    mses = []
    optimizer = torch.optim.LBFGS(model.params, history_size=10, max_iter=5)

    def closure():
        loss, mse, difference, sharp_convolved, blurry_cropped = model.loss(sharp, blurry)

        optimizer.zero_grad()
        loss.backward()
        mses.append(mse.item())

        if len(mses) % 10 == 0:
            print(f"step {len(mses):05d} - MSE {mse.item():20.10f} - loss MSE {loss.item():20.10f}")

        return loss

    while len(mses) < 300:
        optimizer.step(closure)

    # remove return statement to look at PSFs
    return

    with torch.no_grad():
        loss, mse, difference, sharp_convolved, blurry_cropped = model.loss(sharp, blurry)

        nx = 3
        ny = 2
        plt.figure(figsize=(16, 8))
        for i, (title, img, vmin, vmax) in enumerate([
            ("PSF", model.psf, None, None),
            ("sharp_convolved", sharp_convolved, 0, 1),
            ("blurry_cropped", blurry_cropped, 0, 1),
            ("10 $\\times$ difference", 10 * difference, 0, 1),
        ]):
            plt.subplot(ny, nx, 1 + i)
            plt.title(title)
            plt.imshow(img.cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)

        plt.subplot(ny, nx, 5)
        plt.title("MSE")
        plt.semilogy(mses[3:])

        plt.subplot(ny, nx, 6)
        final_psf = model.psf.detach().cpu()
        plt.plot(final_psf[:, final_psf.shape[1] // 2])
        plt.show()

def main():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda")
    sample = 1
    font = "Times"

    psf_radii = {
        0: 15,
        1: 15,
        2: 25,
        3: 35,
        4: 50,
        5: 60,
        6: 70,
        7: 80,
        8: 90,
        9: 110,
        10: 130,
        11: 170,
    }

    initial_offsets = {
        0: 0.135,
        1: 0.135,
        2: 0.2,
        3: 0.3,
        4: 0.33,
        5: 0.4,
        6: 0.4,
        7: 0.4,
        8: 0.4,
        9: 0.4,
        10: 0.4,
        11: 0.4,
    }

    psf_regularizations = {
        0: 1,
        1: 1,
        2: 2,
        3: 5,
        4: 5,
        5: 5,
        6: 10,
        7: 30,
        8: 40,
        9: 80,
        10: 100,
        11: 120,
    }

    for step in range(12):
        print("step", step)
        psf_radius = psf_radii[step]
        initial_offset = initial_offsets[step]
        psf_regularization = psf_regularizations[step]

        model = BlurModel(
            psf_radius=psf_radius,
            psf_regularization=psf_regularization,
            initial_offset=initial_offset,
            M=get_dewarping_matrix(step),
            device=device)

        sharp  = load_hdc(step=step, cam=1, sample=sample, font=font)
        blurry = load_hdc(step=step, cam=2, sample=sample, font=font)

        sharp = torch.tensor(sharp, device=device)
        blurry = torch.tensor(blurry, device=device)

        os.makedirs("psfs", exist_ok=True)

        optimize_model(model, sharp, blurry)

        print("offset:", model.offset)

        path = f"psfs/step_{step}.pth"
        print("save", path)
        torch.save(model, path)
        del model

def test():
    step = 11
    sample = 2
    font = "Times"
    model = torch.load(f"psfs/step_{step}.pth")
    device = torch.device("cuda")

    sharp  = load_hdc(step=step, cam=1, sample=sample, font=font)
    blurry = load_hdc(step=step, cam=2, sample=sample, font=font)

    with torch.no_grad():
        sharp = torch.tensor(sharp, device=device)
        blurry = torch.tensor(blurry, device=device)

        sharp_convolved = model(sharp)

        blurry = model.preprocess(blurry)

        r = model.psf_radius

        difference = torch.abs(sharp_convolved - blurry)

        plt.imshow(model.psf.detach().cpu().numpy(), cmap='gray')
        plt.show()

        for i, (title, img) in enumerate([
            ("sharp", sharp),
            ("blurry", blurry),
            ("sharp_convolved", sharp_convolved),
            ("10 $\\times$ difference", 10 * difference),
        ]):
            plt.subplot(2, 2, 1 + i)
            plt.title(title)
            plt.imshow(img.cpu(), cmap='gray', vmin=0, vmax=1)
        plt.show()



if __name__ == "__main__":
    main()
    test()
