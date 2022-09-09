import numpy as np
from sklearn.model_selection import KFold

from .kerasfeeder import KerasFeeder, SurfaceFeeder

class KFoldKerasFeeder(KerasFeeder):
    """Behaves the same as :obj:`KerasFeeder`, except the training/testing data are split via a KFold approach. Thus, :attr:`features` and :attr:`labels` are lists of dicts, where each element corresponds to a split.

    Note:
        See :attr:`KerasFeeder` for full documentation, this only lists the unique traits.

    Attributes:
        n_splits (int): number of KFold splits, passed to :obj:`sklearn.model_selection.KFold`
        training_fraction (None): not used here
    """
    training_fraction = None
    n_splits = 5

    mask2d = None
    mask1d = None
    active_index = None

    @property
    def n_samples(self):
        if self.labels is not None:
            return len(self.labels[0]["training"][self.sample_dim])
        else:
            return None

    @property
    def x_training(self):
        if self.features is not None:
            return [{k:f["training"][k].data for k in self.feature_names} for f in self.features]
        else:
            return None

    @property
    def x_testing(self):
        if self.features is not None:
            return [{k:f["testing"][k].data for k in self.feature_names} for f in self.features]
        else:
            return None

    def __call__(self, xds):
        xds = self.subset_dataset(xds)
        if self.load_into_memory:
            xds = xds.load();

        # get a 2D mask for unstacking horizontal
        self.mask2d = self.get_mask(xds)
        self.mask1d = self.stack_horizontal(self.mask2d)
        self.active_index = np.arange(len(self.mask1d))[self.mask1d]

        xds = self.stack_horizontal(xds)

        xds = self.remove_nans(xds)

        xds[self.label_name] = self.broadcast_for_label(xds, dimname="member")

        xds = self.stack_sample(xds)

        xds = self.reorder_dims(xds)

        self.set_features_and_labels(xds)

        # here's where it's different...
        if self.normalize_data:
            self.features = [self.normalize(f, self.sample_dim) for f in self.features]

        self.labels = [self.stack_vertical(l) for l in self.labels]

        self.set_keras_inputs(self.features[0]["training"])


    def set_features_and_labels(self, xds, shuffle=True, random_seed=None):
        """Split dataset into :attr:`n_splits` training/testing pairs, and separate features and labels

        Note:
            Validation data is taken from each training dataset. It is not separated here.

        Args:
            xds (:obj:`xarray.Dataset`): with training data and labels
            shuffle (bool, optional): see :obj:`sklearn.model_selection.KFold`
            random_seed (int, optional): RNG seed

        Sets Attributes:
            features (list of dict): each item in list corresponds to each of the KFold splits, containing a typical :obj:`KerasFeeder` dict
            labels (list of dict): similar to features but each item contains :attr:`KerasFeeder` label dict
        """

        kfolder = KFold(n_splits=self.n_splits,
                        shuffle=shuffle,
                        random_state=random_seed
                       )
        features = []
        labels = []
        for train_idx, test_idx in kfolder.split(xds[self.sample_dim]):
            trainer = xds.isel({self.sample_dim : train_idx})
            tester  = xds.isel({self.sample_dim : test_idx })

            features.append(
                {
                    "training"  : {k : trainer[k] for k in self.feature_names},
                    "testing"   : {k : tester[k] for k in self.feature_names},
                }
            )

            labels.append(
                {
                    "training"  : trainer[self.label_name],
                    "testing"   : tester[self.label_name],
                }
            )

            self.features = features
            self.labels = labels


class KFoldSurfaceFeeder(KFoldKerasFeeder, SurfaceFeeder):
    """Behaves the same as :obj:`SurfaceFeeder`, except the training/testing data are split via a KFold approach. Thus, :attr:`features` and :attr:`labels` are lists of dicts, where each element corresponds to a split.
    """
    pass
