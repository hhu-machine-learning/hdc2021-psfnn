from PIL import Image
import numpy as np
import os

def load_hdc(step, cam, sample, font):
    fix_for_inconsistent_naming_scheme = {
        "Times": "timesR",
        "Verdana": "verdanaRef",
    }[font]

    path = f"~/data/hdc2021/step{step}/{font}/CAM{cam:02d}/focusStep_{step}_{fix_for_inconsistent_naming_scheme}_size_30_sample_{sample:04d}.tif"
    path = os.path.expanduser(path)

    image = Image.open(path)

    image = np.float32(image) / 65535.0

    return image
