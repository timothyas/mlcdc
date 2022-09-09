"""
# Hyperparameter optimization

 Our dataset is <= 1GB, but there are many hyperparameters to choose from (learning rate, number of layers, layer
 width, etc.).
 The Hyperband search from (Li et al., 2016) and implemented in dask-ml (Sievert et al., 2018)
 as [HyperbandCV](https://ml.dask.org/modules/generated/dask_ml.model_selection.HyperbandSearchCV.html#dask_ml.model_selection.HyperbandSearchCV)
 is designed to handle this exact case: compute but not memory bound hyperparameter optimization


 “Hyperband: A novel bandit-based approach to hyperparameter optimization”, 2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar. https://arxiv.org/abs/1603.06560

 “Better and faster hyperparameter optimization with Dask”, 2018 by S. Sievert, T. Augspurger, M. Rocklin. https://doi.org/10.25080/Majora-7ddc1dd1-011
 """

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from scipy.stats import uniform, loguniform
from scikeras.wrappers import KerasRegressor
from dask.distributed import Client, get_client, performance_report
from dask_ml.model_selection import HyperbandSearchCV

from mlcdc import SurfaceFeeder

def open_surfacedataset(path):
    """open dataset and compute/select surface quantities"""

    # Preliminary setup
    ds = xr.open_zarr(path)
    ds['log10_atm_tprcp'] = np.log10(ds.atm_tprcp)
    ds['atm_speed_srf'] = ds.atm_wind_speed.isel(alev=-1)
    ds['ocn_speed_srf'] = ds.ocn_speed.isel(olev=0)
    ds['ocn_u_srf'] = ds['ocn_u'].isel(olev=0)
    ds['ocn_v_srf'] = ds['ocn_v'].isel(olev=0)
    ds['log10_atm_tprcp'].attrs['long_name'] = 'derived log10 of tprcp'
    ds['atm_speed_srf'].attrs['long_name'] = 'derived atmospheric surface wind speed'
    ds['ocn_speed_srf'].attrs['long_name'] = 'derived ocean surface current speed'
    ds['ocn_u_srf'].attrs['long_name'] = 'derived zonal ocean surface current speed'
    ds['ocn_v_srf'].attrs['long_name'] = 'derived meridional ocean surface current speed'

    ds['ocn_ubtav'] = ds['ocn_ubtav'].isel(olev=0)
    ds['ocn_vbtav'] = ds['ocn_vbtav'].isel(olev=0)
    ds['ocn_baro_srf'] = np.sqrt(ds.ocn_ubtav**2 + ds.ocn_vbtav**2)
    ds['ocn_baro_srf'].attrs['long_name'] = 'derived barotropic ocean current speed'

    ds['corr_atm_ocn'] = ds.corr_atm_ocn.isel(olev=0,alev=-1)
    return ds


def get_kerasfeeder(ds, feature_names, label_name, mask_name, training_fraction, load_into_memory):

    kf = SurfaceFeeder(feature_names=feature_names,
                       label_name=label_name,
                       mask_name=mask_name,
                       training_fraction=training_fraction,
                       load_into_memory=load_into_memory)


    kf(ds)
    print(" --- Features --- ")
    for key in feature_names:
        if 'long_name' in ds[key].attrs:
            print(f"{key:<24s}: {ds[key].long_name}")

    print()
    print(kf)
    return kf


def build_model(n_layers=1,
                units_per_layer=16,
                hidden_activation=None,
                regularizer=keras.regularizers.L2,
                regularization=1e-7,
               ):
    """Build a keras neural network

    Args:
        n_layers (int, optional): number of layers in NN
        units_per_layer (int, optional): number of nodes  in each layer
        hidden_activation (str, optional): activation function in all of the hidden layers, e.g. "swish", "tanh"
        regularizer (:class:`keras.regularizers.Regularizer`): e.g. L1 or L2
        regularization (float, optional): regularization to be applied to all kernel and bias weights in the network

    Returns:
        model (:obj:`keras.Model`): neural network model
    """

    # hard code this part for now
    ftr = ['atm_q2m', 'atm_qrain', 'atm_t2m','atm_tprcp', 'atm_speed_srf',
           'ocn_MEKE', 'ocn_MLD', 'ocn_sfc', 'ocn_sst', 'ocn_speed_srf', 'ocn_baro_srf']

    inputs = keras.Input(shape=(len(ftr,)), name="all_inputs")

    for i in range(n_layers):
        this_one = inputs if i == 0 else hidden
        hidden = keras.layers.Dense(
            units=units_per_layer,
            activation=hidden_activation,
            kernel_regularizer=regularizer(regularization),
            bias_regularizer=regularizer(regularization),
        )(this_one)

    output = keras.layers.Dense(
        activation='tanh',
        units=1, name='ao_corr'
    )(hidden)

    model = keras.Model(
        inputs=inputs,
        outputs=[output],
    )

    return model


if __name__ == "__main__":


    # Setup data and kerasfeeder
    ds = open_surfacedataset('../data/tcorr.predictors0.10lon.10lat.-1alev.-1olev.-1member.zarr')

    features = [
        'atm_q2m','atm_qrain','atm_t2m','atm_tprcp','atm_speed_srf',
        'ocn_MEKE', 'ocn_MLD', 'ocn_sfc', 'ocn_sst', 'ocn_speed_srf', 'ocn_baro_srf',
    ]

    kf = get_kerasfeeder(ds,
                         feature_names=features,
                         label_name="corr_atm_ocn",
                         mask_name="ocn_mask",
                         training_fraction=0.8,
                         load_into_memory=True)

    client = Client()

    print(" --- Created Client --- ")
    print(client)
    print()

    # HyperbandCV
    # Note: use loss__, optimizer__, or metrics__ to pass kwargs to sub-arguments
    # within model.compile ... easiest to do this and not compile the model in build_model
    params = {
        "n_layers"                  : np.arange(2, 20, 2),
        "units_per_layer"           : [8, 16, 32, 64, 128],
        "hidden_activation"         : ["tanh", "swish", "sigmoid"],
        "regularizer"               : [keras.regularizers.L2,
                                       keras.regularizers.L1],
        "regularization"            : loguniform(1e-9, 1e-2),
        "optimizer__learning_rate"  : loguniform(1e-5, 1e-1),
        "batch_size"                : [16, 32, 64, 128, 256],
        "loss"                      : [keras.losses.MeanAbsoluteError(),
                                       keras.losses.MeanSquaredError(),
                                       keras.losses.Huber(delta=10),
                                       keras.losses.Huber(delta=1),
                                       keras.losses.Huber(delta=0.1)],
        }


    kw = {key : None for key in params.keys()}

    # Note on metrics: it's finicky
    # keras.metrics work with plain KerasRegressor, but not when dask is involved
    # sklearn.metrics never work
    # if it's out of the box anyway ... use the string form
    model = KerasRegressor(build_model,
                           verbose=False,
                           validation_split=0.2,
                           epochs=100,
                           optimizer=keras.optimizers.Adam,
                           metrics=["mean_squared_error"],
                           **kw)

    search = HyperbandSearchCV(estimator=model,
                               parameters=params,
                               max_iter=100,
                               patience=True,
                               tol=1e-2,
                               test_size=0.2,
                               scoring="neg_mean_squared_error",
                               random_state=0)

    X = np.concatenate([val[...,None] for val in kf.x_training.values()], axis=1)
    y = kf.labels['training'].values

    with performance_report(filename="hyperbandcv_report.html"):
        search.fit(X, y)

    best = search.best_estimator_

    pred = best.predict(np.concatenate([val[...,None] for val in kf.x_testing.values()], axis=1))

    fig, ax = plt.subplots()
    ax.scatter(kf.labels['testing'],
               pred,
              )
    ax.plot(kf.labels['testing'],
            kf.labels['testing'],
            color='gray')
    ax.set(xlabel='Truth',ylabel='Prediction')
    fig.savefig("../figures/hyperbandcv_test_data.pdf", bbox_inches="tight")

    print(" --- HyperbandSearchCV Results --- ")
    print("r^2 = ", best.r_squared(kf.labels['testing'].values[...,None], pred[...,None]))

    print()
    print("best params = ", search.best_params_)
