from abc import ABC, abstractmethod
from enum import StrEnum

import einops
import torch
import torch.nn.functional as nnf


class NeighborhoodShape(StrEnum):
	SQUARE = "square"
	CIRCLE = "circle"


class RelativeMagnitudeMode(StrEnum):
	DIFF = "diff"  # RIHOG1
	RATIO = "ratio"  # RIHOG3


def _get_nbhd_masks(
	*,
	win_size: int,
	shape: NeighborhoodShape
):
	c = win_size // 2  # center
	yy, xx = torch.meshgrid(torch.arange(win_size), torch.arange(win_size), indexing="ij")  # (ws ws), (ws ws)
	match shape:
		case NeighborhoodShape.SQUARE:
			cdist = torch.amax(torch.stack([torch.abs(yy - c), torch.abs(xx - c)]), dim=0)  # (ws ws)
		case NeighborhoodShape.CIRCLE:
			cdist = torch.round(  # (ws ws)
				torch.linalg.vector_norm(torch.stack([yy - c, xx - c]).float(), ord=2, dim=0)
			).to(torch.int)
	nbhd_masks = torch.stack([(cdist == dist).to(torch.int) for dist in range(int(cdist.max().item()) + 1)]).bool()  # (nc ws ws)
	return nbhd_masks


def _get_batch_grad(
	batch: torch.Tensor  # (ib ic ih iw)
):
	filt_x = einops.rearrange(torch.tensor([
		[-1, 0, 1],
	], dtype=torch.float, device=batch.device), "h w -> 1 1 h w")  # (1 1 fh fw)
	filt_y = einops.rearrange(torch.tensor([
		[-1],
		[ 0],
		[ 1]
	], dtype=torch.float, device=batch.device), "h w -> 1 1 h w")  # (1 1 fh fw)
	batch_grad_x = torch.conv2d(nnf.pad(batch, pad=(1,1,0,0), mode="replicate"), filt_x)  # (ib 1 ih iw)
	batch_grad_y = torch.conv2d(nnf.pad(batch, pad=(0,0,1,1), mode="replicate"), filt_y)  # (ib 1 ih iw)
	batch_grad_magn = torch.sqrt(batch_grad_x.square() + batch_grad_y.square())  # (ib 1 ih iw)
	batch_grad_angl = torch.atan2(batch_grad_y, batch_grad_x)  # (ib 1 ih iw)
	batch_grad = torch.cat([batch_grad_magn, batch_grad_angl], dim=1)  # (ib ig ih iw)
	return batch_grad


class RIHOGBase(ABC):
	def __init__(
		self,
		nbhd_steps: int,
		num_bins: int,
		nbhd_shape: NeighborhoodShape = NeighborhoodShape.SQUARE,
		rel_magn_mode: RelativeMagnitudeMode = RelativeMagnitudeMode.RATIO
	):
		assert isinstance(nbhd_steps, int) and nbhd_steps > 0, "'nbhd_steps' must be a positive integer"
		assert isinstance(num_bins, int) and num_bins > 0, "`num_bins` must be a positive integer"
		assert nbhd_shape in NeighborhoodShape, f"'nbhd_shape' must be one of {list(map(str, NeighborhoodShape))}"
		assert rel_magn_mode in RelativeMagnitudeMode, f"'rel_magn_mode' must be one of {list(map(str, RelativeMagnitudeMode))}"
		self.nbhd_steps = nbhd_steps
		self.num_bins = num_bins
		self.nbhd_shape = nbhd_shape
		self.rel_magn_mode = rel_magn_mode
	
	def __repr__(self):
		return f"{self.__class__.__name__}(\n" \
			f"    nbhd_steps={self.nbhd_steps!r},\n" \
			f"    num_bins={self.num_bins!r},\n" \
			f"    nbhd_shape={self.nbhd_shape.value!r},\n" \
			f"    rel_magn_mode={self.rel_magn_mode.value!r}\n" \
			")"
	
	@property
	def window_size(self):
		return 1 + 2 * self.nbhd_steps
	
	@abstractmethod
	def _compute(
		self,
		b_win_grad_magn: torch.Tensor,  # (ib wc ws ws)
		b_win_grad_angl: torch.Tensor,  # (ib wc ws ws)
		nbhd_masks: torch.Tensor,  # (nc ws ws)
		IB: int,
		WC: int,
		NC: int
	) -> torch.Tensor:
		...
	
	def __call__(
		self,
		batch: torch.Tensor,  # (ib ic ih iw)
	):
		"""
		Args:
			batch:
				Tensor of size `(b 1 h w)`.
		
		Returns:
			Tensor of size `(b wc nc*bc)`, where
			- `wc`: window count
			- `nc`: neighborhood count
			- `bc`: bin count
		
		Raises:
			AssertionError:
				The arguments are not valid.
		"""
		assert isinstance(batch, torch.Tensor) and batch.dim() == 4, "input batch must be a 4-dimensional tensor"
		IB, IC, IH, IW = batch.shape
		assert IC == 1, "input batch must have one and only one channel"
		## compute neighborhoods masks
		WS = self.window_size  # window size
		assert IH >= WS and IW >= WS, "spatial dimensions of input batch must be of equal or larger size than the largest neighborhood window"
		nbhd_masks = _get_nbhd_masks(win_size=WS, shape=self.nbhd_shape)  # (nc ws ws)
		NC = nbhd_masks.shape[-3]  # neighborhoods count
		## compute windowed gradient
		b_grad = _get_batch_grad(batch)  # (ib ig ih iw)
		b_win_grad = einops.rearrange(  # (ib wc ig ws ws), wc: window count
			nnf.unfold(b_grad, kernel_size=WS),
			"ib (ig wh ww) (nvw nhw) -> ib (nvw nhw) ig wh ww",
			ig=2, wh=WS, ww=WS, nvw=(IH - WS + 1), nhw=(IW - WS + 1)
		)
		WC = b_win_grad.shape[-4]
		b_win_grad_magn, b_win_grad_angl = b_win_grad[..., 0, :, :], b_win_grad[..., 1, :, :]  # (ib wc ws ws), (ib wc ws ws)
		return self._compute(
			b_win_grad_magn,
			b_win_grad_angl,
			nbhd_masks,
			IB,
			WC,
			NC
		)


class RIHOG(RIHOGBase):
	"""Vectorized RIHOG implementation.
	
	Computes the Rotation Invariant Histogram of Oriented Gradients as described in M. Cheon et al., "Rotation Invariant
	Histogram of Oriented Gradients", International Journal of Fuzzy Logic and Intelligent Systems, vol. 11, no. 4, December
	2011 (https://doi.org/10.5391/IJFIS.2011.11.4.293).
	
	Attributes:
		nbhd_steps:
			The number of histograms to compute for each pixel, one per neighborhood step.
		num_bins:
			The number of histogram bins.
		nbhd_shape:
			The shape of the neighborhood. :py:attr:`rihog.NeighborhoodShape.SQUARE` matches the original description.
		rel_magn_mode:
			How to compute the relative magnitude. :py:attr:`rihog.RelativeMagnitudeMode.DIFF` matches the original description
			of RIHOG1, while :py:attr:`rihog.RelativeMagnitudeMode.RATIO` is equivalent to RIHOG3.
	
	Raises:
			AssertionError:
				The arguments are not valid.
	"""
	
	def _compute(
		self,
		b_win_grad_magn: torch.Tensor,  # (ib wc ws ws)
		b_win_grad_angl: torch.Tensor,  # (ib wc ws ws)
		nbhd_masks: torch.Tensor,  # (nc ws ws)
		IB: int,
		WC: int,
		NC: int
	):
		## compute relative orientation
		b_win_grad_angl = einops.repeat(b_win_grad_angl, "ib wc wh ww -> ib wc nc wh ww", nc=NC)  # (ib wc nc ws ws)
		# grad angle for all neighborhoods concatenated one after the other in the last dim
		b_win_flat_nbhd_grad_angl = b_win_grad_angl[:, :, nbhd_masks]  # (ib wc nc*ns), ns: neighborhood size
		b_win_flat_nbhd_rel_grad_angl = b_win_flat_nbhd_grad_angl[..., 0:1] - b_win_flat_nbhd_grad_angl  # (ib wc nc*ns)
		b_win_flat_nbhd_rel_grad_angl[b_win_flat_nbhd_rel_grad_angl < -torch.pi] = \
			(b_win_flat_nbhd_rel_grad_angl + 2 * torch.pi)[b_win_flat_nbhd_rel_grad_angl < -torch.pi]
		b_win_flat_nbhd_rel_grad_angl[b_win_flat_nbhd_rel_grad_angl > torch.pi] = \
			(b_win_flat_nbhd_rel_grad_angl - 2 * torch.pi)[b_win_flat_nbhd_rel_grad_angl > torch.pi]
		## bucketize relative orientation
		# bucketize gives idx 0 to anything smaller than first boundary
		# discard first boundary so anything smaller than the second one gets idx 0, avoiding numerical errors near the boundaries
		# same for last and second-to-last boundaries
		bin_bndrs = torch.linspace(-torch.pi, torch.pi, steps=self.num_bins+1, device=b_win_flat_nbhd_rel_grad_angl.device)[1:-1]
		b_win_flat_nbhd_bucket_idx = torch.bucketize(b_win_flat_nbhd_rel_grad_angl, boundaries=bin_bndrs)  # (ib wc nc*ns)
		## split relative orientation for each neighborhood
		nbhd_size = nbhd_masks.sum(dim=(-2,-1))  # (nc)
		nbhd_b_win_bucket_idx_tup = b_win_flat_nbhd_bucket_idx.split(nbhd_size.tolist(), dim=-1)  # nc * (ib wc ns)
		## compute relative magnitude and split for each neighborhood
		b_win_grad_magn = einops.repeat(b_win_grad_magn, "ib wc wh ww -> ib wc nc wh ww", nc=NC)  # (ib wc nc ws ws)
		b_win_flat_nbhd_grad_magn = b_win_grad_magn[:, :, nbhd_masks]  # (ib wc nc*ns)
		match self.rel_magn_mode:
			case RelativeMagnitudeMode.DIFF:
				b_win_flat_nbhd_rel_grad_magn = torch.abs(b_win_flat_nbhd_grad_magn[..., 0:1] - b_win_flat_nbhd_grad_magn)  # (ib wc nc*ns)
			case RelativeMagnitudeMode.RATIO:
				b_win_flat_nbhd_rel_grad_magn = torch.atan(b_win_flat_nbhd_grad_magn[..., 0:1] / (1e-5 + b_win_flat_nbhd_grad_magn))  # (ib wc nc*ns)
		nbhd_b_win_rel_grad_magn_tup = b_win_flat_nbhd_rel_grad_magn.split(nbhd_size.tolist(), dim=-1)  # nc * (ib wc ns)
		## compute histogram
		nbhd_b_win_grad_hist_list = []
		for (
			nbhd_b_win_rel_grad_magn,  # (ib wc ns)
			nbhd_b_win_bucket_idx  # (ib wc ns)
		) in zip(nbhd_b_win_rel_grad_magn_tup[1:], nbhd_b_win_bucket_idx_tup[1:]):  # [1:]: discard center of window
			nbhd_b_win_grad_hist = torch.zeros((IB, WC, self.num_bins), device=nbhd_b_win_rel_grad_magn.device)  # (ib wc bc), bc: bin count
			nbhd_b_win_grad_hist.scatter_add_(src=nbhd_b_win_rel_grad_magn, index=nbhd_b_win_bucket_idx, dim=-1)
			nbhd_b_win_grad_hist = nnf.normalize(nbhd_b_win_grad_hist, dim=-1)
			nbhd_b_win_grad_hist_list.append(nbhd_b_win_grad_hist)
		rihog = torch.cat(nbhd_b_win_grad_hist_list, dim=-1)  # (ib wc nc*bc)
		return rihog


class NaiveRIHOG(RIHOGBase):
	"""Non-vectorized RIHOG implementation. Refer to :py:class:`rihog.RIHOG` for more information."""
	
	def _compute(
		self,
		b_win_grad_magn: torch.Tensor,  # (ib wc ws ws)
		b_win_grad_angl: torch.Tensor,  # (ib wc ws ws)
		nbhd_masks: torch.Tensor,  # (nc ws ws)
		IB: int,
		WC: int,
		NC: int
	):
		nbhd_b_win_grad_hist_list = []
		for nbhd_mask in nbhd_masks[1:]:  # (ws ws)
			nbhd_b_win_rel_grad_angl = b_win_grad_angl[..., nbhd_masks[0]] - b_win_grad_angl[..., nbhd_mask]  # (ib wc ns)
			nbhd_b_win_rel_grad_angl[nbhd_b_win_rel_grad_angl < -torch.pi] = \
				(nbhd_b_win_rel_grad_angl + 2 * torch.pi)[nbhd_b_win_rel_grad_angl < -torch.pi]
			nbhd_b_win_rel_grad_angl[nbhd_b_win_rel_grad_angl > torch.pi] = \
				(nbhd_b_win_rel_grad_angl - 2 * torch.pi)[nbhd_b_win_rel_grad_angl > torch.pi]
			match self.rel_magn_mode:
				case RelativeMagnitudeMode.DIFF:
					nbhd_b_win_rel_grad_magn = torch.abs(  # (ib wc ns)
						b_win_grad_magn[..., nbhd_masks[0]] - b_win_grad_magn[..., nbhd_mask])
				case RelativeMagnitudeMode.RATIO:
					nbhd_b_win_rel_grad_magn = torch.atan(  # (ib wc ns)
						b_win_grad_magn[..., nbhd_masks[0]] / (1e-5 + b_win_grad_magn[..., nbhd_mask]))
			nbhd_img_win_grad_hist_list = []
			for nbhd_img_win_rel_grad_angl, nbhd_img_win_rel_grad_magn in zip(nbhd_b_win_rel_grad_angl, nbhd_b_win_rel_grad_magn):  # (wc ns), (wc ns)
				nbhd_win_grad_hist_list = []
				for nbhd_win_rel_grad_angl, nbhd_win_rel_grad_magn in zip(nbhd_img_win_rel_grad_angl, nbhd_img_win_rel_grad_magn):  # (ns), (ns)
					nbhd_win_grad_hist = nbhd_win_rel_grad_angl.histogram(  # (bc)
						bins=self.num_bins, range=(-torch.pi, torch.pi),
						weight=nbhd_win_rel_grad_magn
					).hist
					nbhd_win_grad_hist = nnf.normalize(nbhd_win_grad_hist, dim=0)
					nbhd_win_grad_hist_list.append(nbhd_win_grad_hist)
				nbhd_img_win_grad_hist = torch.stack(nbhd_win_grad_hist_list)  # (wc bc)
				nbhd_img_win_grad_hist_list.append(nbhd_img_win_grad_hist)
			nbhd_b_win_grad_hist = torch.stack(nbhd_img_win_grad_hist_list)  # (ib wc bc)
			nbhd_b_win_grad_hist_list.append(nbhd_b_win_grad_hist)
		rihog = torch.cat(nbhd_b_win_grad_hist_list, dim=-1)  # (ib wc nc*bc)
		return rihog


def rihog(
	batch: torch.Tensor,
	*,
	nbhd_steps: int,
	num_bins: int,
	nbhd_shape: NeighborhoodShape = NeighborhoodShape.SQUARE,
	rel_magn_mode: RelativeMagnitudeMode = RelativeMagnitudeMode.RATIO
):
	"""Computes the RIHOG in a vectorized fashion. Refer to :py:class:`rihog.RIHOG` for more information."""
	return RIHOG(
		nbhd_steps=nbhd_steps,
		num_bins=num_bins,
		nbhd_shape=nbhd_shape,
		rel_magn_mode=rel_magn_mode
	)(batch=batch)


def rihog_naive(
	batch: torch.Tensor,
	*,
	nbhd_steps: int,
	num_bins: int,
	nbhd_shape: NeighborhoodShape = NeighborhoodShape.SQUARE,
	rel_magn_mode: RelativeMagnitudeMode = RelativeMagnitudeMode.RATIO
):
	"""Computes the RIHOG in a non-vectorized fashion. Refer to :py:class:`rihog.RIHOG` for more information."""
	return NaiveRIHOG(
		nbhd_steps=nbhd_steps,
		num_bins=num_bins,
		nbhd_shape=nbhd_shape,
		rel_magn_mode=rel_magn_mode
	)(batch=batch)
