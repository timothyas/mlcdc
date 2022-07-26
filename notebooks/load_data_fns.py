import re
import numpy as np
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
  dsnew = ds.copy()
  # File name contains ensemble member index
  filename = dsnew.filename
  filenum = get_filenum(filename)
  # Add ens_mem dimension
  dsnew['ens_mem'] = (filenum - 1) # Files are indexed from 1
  dsnew = dsnew.expand_dims('ens_mem').set_coords('ens_mem')
  return dsnew  
