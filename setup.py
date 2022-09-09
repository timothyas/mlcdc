from setuptools import setup

install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "zarr",
        "tensorflow",
]

setup(
        name="mlcdc",
        description="Machine Learning for estimating Cross Domain Correlations",
        url="https://github.com/noaa-psd/mlcdc",
        author="Zofia Stanely & Timothy Smith",
        install_requires=install_requires,
)
