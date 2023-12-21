# import root_pandas
import numpy as np
import pandas as pd
from decorrelator import cdfCalc
import argparse
from coffea.nanoevents import NanoEventsFactory, BaseSchema
import os

def main(options):

    df = pd.DataFrame()

    import glob
    #files = glob.glob( str(options.infile) + "*.parquet")
    #data = [pd.read_parquet(f) for f in files]
    #data = pd.read_parquet("/net/scratch_cms3a/daumann/massresdecorrhiggsdna/big_bkg/Diphoton.parquet")
    #events= pd.concat(data,ignore_index=True)

    files  = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/diphoton_samples_for_CDFs/Diphoton_2022_postEE/nominal/*.parquet" )
    data   = [pd.read_parquet(f) for f in files]
    events = pd.concat(data,ignore_index=True)

    df["sigma_m_over_m"] = events.sigma_m_over_m_Smeared.to_numpy()
    df["mass"]   = events.mass.to_numpy()
    df["weight"] = events.weight.to_numpy()    

    print(f"INFO: found {len(events)} events")

    # clarify with anyone what this "sigmarv" business here is about!
    df["sigma_m_over_m"] = events.sigma_m_over_m_Smeared.to_numpy()

    df["mass"]   = events.mass.to_numpy()
    df["weight"] = events.weight.to_numpy()

    # df = root_pandas.read_root(options.infile, options.tree, columns=['sigmarv', 'recoMass', 'weight'])
    calc = cdfCalc(df, 'sigma_m_over_m', 'mass', np.linspace(100, 180, 161)) # 161 bins initially
    calc.dumpCdfs(options.cdfsFile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    #requiredArgs.add_argument('-i', '--infile', action='store', type=str, required=True)
    requiredArgs.add_argument('-i', '--infile', action='store', type=str, required=True)
    requiredArgs.add_argument('-c', '--cdfsFile', action='store', type=str, required=True)
    requiredArgs.add_argument('-t', '--tree', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
