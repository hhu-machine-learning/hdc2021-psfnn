import torch
import torch.nn.functional as F
import torch.fft
from dewarp import warp

def fft_convolve(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape

    assert(image.shape[0] >= kernel.shape[0])
    assert(image.shape[1] >= kernel.shape[1])

    kernel = F.pad(kernel, (0, iw - kw, 0, ih - kh))

    x = torch.fft.rfftn(image)
    y = torch.fft.rfftn(kernel)
    z = x * y
    z = torch.fft.irfftn(z, s=(ih, iw))
    z = z[kh - 1:, kw - 1:]
    z = z[:ih - kh + 1, :iw - kw + 1]
    return z

class BlurModel(torch.nn.Module):
    def __init__(self, psf_radius, psf_regularization, initial_offset, M, device):
        psf_size = 2 * psf_radius + 1

        M = torch.tensor(M, device=device)

        psf = torch.ones((psf_size, psf_size), device=device) / psf_size**2

        offset = torch.tensor([initial_offset], device=device)

        params = [psf, offset, M]

        for param in params:
            param.requires_grad = True

        self.psf_regularization = psf_regularization
        self.psf_radius = psf_radius
        self.params = params
        self.psf = psf
        self.offset = offset
        self.M = M

    def __call__(self, sharp):
        return fft_convolve(sharp, self.psf) + self.offset

    def preprocess(self, blurry):
        psf_radius = self.psf_radius

        blurry = warp(blurry, self.M)

        return blurry[psf_radius:-psf_radius, psf_radius:-psf_radius]

    def loss(self, sharp, blurry):
        psf = self.psf

        sharp_convolved = self(sharp)

        blurry_preprocessed = self.preprocess(blurry)

        difference = sharp_convolved - blurry_preprocessed

        mse = torch.mean(torch.square(difference))

        loss = mse + self.psf_regularization * torch.mean(torch.abs(psf))

        return loss, mse, difference, sharp_convolved, blurry_preprocessed

