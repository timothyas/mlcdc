
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    import mlcdc

    gdc = mlcdc.GCMDataConverter(zstore_dir="/work2/noaa/gsienkf/tsmith/mlcdc/data")
    gdc()
