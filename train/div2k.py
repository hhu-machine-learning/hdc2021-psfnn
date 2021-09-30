import os, io, zipfile
from PIL import Image
import numpy as np

class Dataset:
    def __init__(self, train=True):
        # Download dataset from:
        # https://data.vision.ee.ethz.ch/cvl/DIV2K/

        if train:
            path = "~/data/DIV2K_train_HR.zip"
        else:
            path = "~/data/DIV2K_valid_HR.zip"

        path = os.path.expanduser(path)

        z = zipfile.ZipFile(path)

        paths = []
        for info in z.infolist():
            if info.is_dir(): continue

            assert(info.filename.endswith(".png"))

            paths.append(info.filename)

        self.paths = paths
        self.z = z

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        path = self.paths[idx]

        data = self.z.read(path)

        # Convert image file bytes to numpy array (probably of type uint8)
        image = np.array(Image.open(io.BytesIO(data)))

        return image

def test():
    for train in [False, True]:
        for image in Dataset(train=train):
            print(image.shape)

if __name__ == "__main__":
    test()
