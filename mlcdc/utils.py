import numpy as np


def split_dataset(xds, dim, fraction, random_seed=None):
    """Split a dataset in 2 by fraction, along a specific dimension. Split using a random set of indices

    Args:
        xds (:obj:`xarray.Dataset`): dataset to be split
        dim (str): dimension to do the splitting along
        fraction (float): how to partition datasets
        random_seed (int): RNG seed

    Returns:
        xds1, xds2 (:obj:`xarray.Dataset`): split datasets, first has ``fraction`` of the original data
    """
    indices = random_indices(n_size=len(xds[dim]),
                             n_indices=int(len(xds[dim])*fraction),
                             random_seed=random_seed)

    all_indices = np.arange(len(xds[dim]))
    others = np.array([i for i in all_indices if i not in indices])
    xds1 = xds.isel({dim:indices})
    xds2 = xds.isel({dim:others})
    return xds1, xds2


def random_indices(n_size, n_indices, random_seed=None):
    """Return random logical indices corresponding to a 1D array with size ``n_size``, returning an array with ``n_indices``.

    Args:
        n_size (int): total number to sample from (0->``n_size``)
        n_indices (int): how many indices to return
        random_seed (int, optional): RNG seed

    Returns:
        indices (array_like): random indices corresponding to arr
    """

    rstate = np.random.RandomState(random_seed)
    return rstate.choice(n_size,
                         size=n_indices,
                         replace=False)


class XNormalizer():
    """Simple class to apply normalization to xarray datasets, using notation from tensorflow.keras. For example, to obtain a normalization operation and apply it to a test dataset:

    xn = XNormalizer(dims='sample')
    xn.adapt( trainer )
    xn( tester )

    Attributes:
        mean, std (:obj:`xarray.Dataset`): mean and standard deviation computed in :meth:`adapt`
    """
    mean = None
    std = None
    def __init__(self, dims="",):
        """
        Args:
            dims (str or list of str): dimension to apply operation along
        """
        self.dims = dims


    def __call__(self, arr):
        """Apply standard normalization: (arr - mean) / std
        using :attr:`mean` and :attr:`std`: set by :meth:`adapt`

        Returns:
            xda (:obj:`xarray.DataArray`): normalized array
        """
        return (arr - self.mean) / self.std


    def adapt(self, arr):
        """Compute mean and std, use this to normalize other arrays

        Sets Attributes:
            mean, std (:obj:`xarray.DataArray`): arr.mean(self.dims) and arr.std(self.dims)
        """
        self.mean = arr.mean(self.dims)
        self.std = arr.std(self.dims)
