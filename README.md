# mlcdc
Machine Learning for estimating Cross Domain Covariance (&amp; Correlation) relationships in coupled atmosphere-ocean DA systems

## General TODO

### Model related

- [ ] Want activation function relevant to correlation (-1, 1)
- [ ] How to choose:
    - regularization type and parameter
    - learning rate
- [ ] Eventually: how to do with xarray/dask? Use tf data loader?


### Data related

- [ ] Split scripts in notebooks to separate dir
- [ ] Organize data preprocessing pipeline to be more portable
    - [ ] First, take care of:
        1. masking
        2. splitting training, validation, and testing
        3. dimension flattening
    - [ ] Eventually, pipeline leading up to this stage
