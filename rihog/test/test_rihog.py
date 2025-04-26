import torch

from ..rihog import NeighborhoodShape, _get_nbhd_masks, rihog, rihog_naive


def test_square_nbhd():
	assert torch.all(_get_nbhd_masks(win_size=5, shape=NeighborhoodShape.SQUARE) == torch.tensor([
		[[0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0],
		 [0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0]],
		[[0, 0, 0, 0, 0],
		 [0, 1, 1, 1, 0],
		 [0, 1, 0, 1, 0],
		 [0, 1, 1, 1, 0],
		 [0, 0, 0, 0, 0]],
		[[1, 1, 1, 1, 1],
		 [1, 0, 0, 0, 1],
		 [1, 0, 0, 0, 1],
		 [1, 0, 0, 0, 1],
		 [1, 1, 1, 1, 1]]]).bool())


def test_circle_nbhd():
	assert torch.all(_get_nbhd_masks(win_size=5, shape=NeighborhoodShape.CIRCLE) == torch.tensor([
		[[0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0],
		 [0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0]],
		[[0, 0, 0, 0, 0],
		 [0, 1, 1, 1, 0],
		 [0, 1, 0, 1, 0],
		 [0, 1, 1, 1, 0],
		 [0, 0, 0, 0, 0]],
		[[0, 1, 1, 1, 0],
		 [1, 0, 0, 0, 1],
		 [1, 0, 0, 0, 1],
		 [1, 0, 0, 0, 1],
		 [0, 1, 1, 1, 0]],
		[[1, 0, 0, 0, 1],
		 [0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0],
		 [1, 0, 0, 0, 1]]]).bool())


def test_naive_vectorized():
	IN_SHAPE = (2, 1, 6, 9)
	NBHD_STEPS = 2
	NUM_BINS = 8
	OUT_SHAPE = (2, 10, 16)
	torch.manual_seed(0)
	b = torch.rand(IN_SHAPE)
	r = rihog(batch=b, nbhd_steps=NBHD_STEPS, num_bins=NUM_BINS)
	assert r.shape == OUT_SHAPE
	rn = rihog_naive(batch=b, nbhd_steps=NBHD_STEPS, num_bins=NUM_BINS)
	assert rn.shape == OUT_SHAPE
	assert torch.allclose(r, rn)
