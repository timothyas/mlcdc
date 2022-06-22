
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import re
from scipy import stats

def get_filename(num):
  path = '/work/noaa/gsienkf/zstanley/projects/obs_loc/data/'
  filenum = f"{num+1:02}"
  filename = 'ens1_0000'+filenum+'.nc'
  return path+filename

def get_filenum(filename):
  match_obj = re.search('(?<=ens1_0000).*(?=\.nc)', filename)
  filenum = int(match_obj.group(0))
  return filenum


def preprocess(ds):
  ''' Add an ensemble member coordinate'''
  filenum = get_filenum(ds.filename)
  return ds.expand_dims({'ens_mem': [filenum-1]})


if __name__ == "__main__":

    ## Where we are working
    data_dir = '/work2/noaa/gsienkf/weihuang/WCLEKF_PRODFORECAST/20151205000000/latlongrid-20151206.030000/AtmOcnIce'

    ## Open netcdf files with xarray
    ds = xr.open_mfdataset(data_dir+'/ens1*.nc', autoclose=True, preprocess=preprocess, parallel=True)

    ds["atm_z"] = ds["atm_DZ"].cumsum("atm_lev")

    # subsample
    ds = ds.rename({"ocn_Temp": "ocn_T"})
    ds = ds[["atm_T", "ocn_T", "atm_lev"]]
    spot = {"lon": 240.5, "lat": 0.5}
    ds = ds.sel(spot)
    ds = ds.swap_dims({"atm_lev": "atm_z"})

    # convert atm_T to K
    ds["atm_T"] -= 273.15
    ds["atm_T"].attrs["units"] = "degC"

    print("ds: ", ds)
    print()
    print("ocnT: ", ds.ocn_T)
    print()
    print("atmT: ", ds.atm_T)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12,10), constrained_layout=True)

    for dom, ax in zip(["atm", "ocn"], axs):
        xda = ds[f"{dom}_T"]
        zdim = ds[f"{dom}_lev"]
        avg = xda.mean("ens_mem")
        std = xda.std("ens_mem")

        label = dom[0].upper()+dom[1:]
        p = ax.plot(avg, zdim, label=label)
        ax.fill_betweenx(zdim, avg-std, avg+std, color=p[0].get_color(), alpha=.3)

        ax.legend()
        ax.set(ylabel=label + " Level", xlabel=r"Temperature ($^\circ$C)")
        ax.invert_yaxis()

    fig.savefig("../figures/hello_zworld.pdf", bbox_inches='tight')
