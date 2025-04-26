# rihog <a href="https://pypi.org/project/rihog"><img src="https://img.shields.io/pypi/v/rihog.svg?style=flat-square"/></a>

Vectorized PyTorch implementation of *[M. Cheon et al., Rotation Invariant Histogram of Oriented Gradients, International Journal of Fuzzy Logic and Intelligent Systems, vol. 11, no. 4, December 2011](https://doi.org/10.5391/IJFIS.2011.11.4.293)*.

<p align="center">
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/A9eazdbg_81c239_44c.jpg" width="50%"/>
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/A914ka8p9_81c23c_44c.jpg" width="50%"/>
</p>

## Install

```
pip install rihog
```

## Usage

``` python
import rihog
import torch

img_b: torch.Tensor  # shape: (b 1 h w)

# class-based interface
rihog_fextr = rihog.RIHOG(...)
img_rihog_b = rihog_fextr(img_b)  # shape: (b wc nc*bc)
# wc: window count, nc: neighborhood count, bc: bin count

# functional interface
img_rihog_b = rihog.rihog(img_b, ...)  # shape: (b wc nc*bc)
```

## Benchmark

<p align="center">
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/original_brodatz.png"/>
</p>

Comparison of RIHOG1 and RIHOG3 against HOG and SIFT on the [Original Brodatz Texture database](https://multibandtexture.recherche.usherbrooke.ca/original_brodatz.html). All algorithms are evaluated on the cosine distance-weighted k-NN accuracy computed from the average feature of each image, across three sizes of the feature vector (`dimension`). Following the evaluation procedure from the paper, 12 rotated versions are generated for each image (30° steps from 0° to 330°) and split into training set (first N rotations) and testing set (remaining rotations).

All benchmarks are run on the following hardware:
- CPU: i9-11900KF @ 3.50GHz
- RAM: 64GB @ 2400Mhz CL 17
- GPU: RTX 3090 24GiB

<p align="center">
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/comp__train=1.png"/>
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/comp__train=2.png"/>
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/comp__train=3.png"/>
</p>

Following is a run time comparison of the vectorized implementation on CPU and CUDA against the naive (non-vectorized) implementation across three hyper-parameter spaces: number of neighborhoods computed for each pixel (`nbhd_steps` parameter), batch size and image size.

<p align="center">
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/bench__nbhd_steps.png"/>
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/bench__batch_size.png"/>
	<img src="https://raw.githubusercontent.com/DSAureli/rihog/refs/heads/main/res/bench__image_size.png"/>
</p>

## License

[MIT](LICENSE)
