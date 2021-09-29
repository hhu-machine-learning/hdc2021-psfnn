# TODO

installation instructions
requirements
usage instructions
example images

# hdc2021-psfnn
Deblur images of the HDC2021 dataset.

* [Challenge Website: https://www.fips.fi/HDCdata.php#anchor1](https://www.fips.fi/HDCdata.php#anchor1)
* [HDC2021 Dataset: https://zenodo.org/record/4916176](https://zenodo.org/record/4916176)
* [DIV2k Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

# Algorithm

For each stage/step:

1. Estimate a coordinate transformation matrix to dewarp the blurry images so they match the sharp images better
2. Estimate a point spread function and an offset to minimize the mean squared error between the convolved sharp image plus offset and the dewarped blurry images. Use regularization to prevent PSF from blowing up.
3. Blur a bunch of natural images from DIV2K dataset with estimated PSF.
4. Jointly train a neural network to deblur images from both the DIV2K dataset and the HDC2021 dataset.

# Citations

The neural network used here is adapted (almost entirely copied) from [https://github.com/MarcoForte/FBA_Matting](F, B, Alpha Matting)

```
@article{forte2020fbamatting,
  title   = {F, B, Alpha Matting},
  author  = {Marco Forte and François Pitié},
  journal = {CoRR},
  volume  = {abs/2003.07711},
  year    = {2020},
}
```

NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study

* https://people.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf
* https://data.vision.ee.ethz.ch/cvl/DIV2K/

```
@InProceedings{Agustsson_2017_CVPR_Workshops,
	author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
}
```

# Authors

* Thomas Germer
* Tobias Uelwer
* Stefan Harmeling

(All affiliated with Heinrich Heine University Düsseldorf)
