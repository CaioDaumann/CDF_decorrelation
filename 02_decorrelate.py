import decorrelator as decorr
import uproot
from coffea.nanoevents import NanoEventsFactory, BaseSchema
import awkward as ak
# import root_numpy
# import ROOT
import argparse
import os
import pandas
import numpy as np
import time
import copy
import pyarrow.parquet as pq


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = chr(9608), printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()

def diphoton_ak_array(diphotons: ak.Array) -> ak.Array:

    output = {}
    for field in ak.fields(diphotons):
        # prefix = self.prefixes.get(field, "")
        # if len(prefix) > 0:
        #     for subfield in ak.fields(diphotons[field]):
        #         if subfield != "__systematics__":
        #             output[f"{prefix}_{subfield}"] = diphotons[field][subfield]
        # else:
        output[field] = diphotons[field]
    return ak.Array(output)


def getArrayBranchName(branchname, fieldname, index):
    if index != ():
        return '{}{}'.format(branchname, index[0])
    return '{}'.format(branchname)

def _main(options):

    if options.inFile == options.outFile:
        raise RuntimeError('Outfile will be recreated, cannot be the same as infile')

    if os.path.exists(options.outFile):
        print("WARNING: outfile exists.")
        # os.remove(options.outFile)

    oFile = ROOT.TFile.Open(options.outFile, 'RECREATE')
    openDir = [oFile]

    cols = options.columns
    totalItems = len(options.tree) * len(options.inFile)
    count = 0
    missingTrees = []

    dummyDf = pandas.DataFrame({'{}'.format(options.var): [0], '{}'.format(options.dVar): [0]})
    decl = decorr.decorrelator(dummyDf, options.var, options.dVar, np.linspace(100., 180., 161))
    decl.loadCdfs(options.cdfFile)

    start = time.time()
    for fle in options.inFile:
        with uproot.open(fle) as f:
            for tree in options.tree:
                # print('Loading {} from {}'.format(tree, fle))
                colsHere = copy.deepcopy(cols)
                if tree == options.tree[0] and options.nomColumns is not None:
                    colsHere += options.nomColumns
                try:
                    df = f[tree].pandas.df(colsHere, flatname=getArrayBranchName)
                except KeyError as e:
                    missingTrees.append((fle, tree))
                    print("KeyError: {}".format(e))
                    count += 1
                    continue

                decl.df = df.loc[:, [options.var, options.dVar]]
                decl.df.reset_index(inplace=True)
                df['{}_decorr'.format(options.var)] = decl.doDecorr(options.ref)
            
                if 'sigmaMoM_decorr' in df.columns:
                    df['sigmaMoM_decorrOld'] = df['sigmaMoM_decorr']
            
                if options.var == 'sigmarv':
                    df['sigmaMoM_decorr'] = df['sigmarv_decorr']

                if options.var == 'sigmaRV':
                    df['sigmaMoM_decorr'] = df['sigmaRV_decorr']
        
                # df.to_root(options.outFile, key=tree, mode='a')
                reqDir = tree.split('/')[:-1]
                if not openDir == reqDir:
                    oFile.cd()
                    openDir = [oFile]
                    for dirName in reqDir:
                        oDir = openDir[-1].Get(dirName)
                        if not oDir:
                            oDir = openDir[-1].mkdir(dirName)
                        oDir.cd()
                        openDir.append(oDir)

                key = tree.split('/')[-1]
                exTree = openDir[-1].Get(key)
                if not exTree:
                    exTree = None
                ttree = root_numpy.array2tree(df.to_records(index=False), name=key, tree=exTree)
                ttree.Write(key, ROOT.TFile.kOverwrite)
                del ttree, df
            
                printProgressBar(count + 1, totalItems, prefix = 'Progress:', suffix = 'Complete')
                count += 1

    end = time.time()
    print('Time needed: {0:.0f} s'.format(end - start))
    print('Trees missing: {}'.format(missingTrees))
    oFile.Close()


def main(options):

    if options.infile == options.outFile:
        raise RuntimeError('Outfile will be recreated, cannot be the same as infile')

    if os.path.exists(options.outFile):
        print("WARNING: outfile exists.")
        # os.remove(options.outFile)

    cols = options.columns
    totalItems = len(options.tree) * len(options.infile)
    count = 0
    missingTrees = []

    dummyDf = pandas.DataFrame({'{}'.format(options.var): [0], '{}'.format(options.dVar): [0]})
    decl = decorr.decorrelator(dummyDf, options.var, options.dVar, np.linspace(100., 180., 161)) #161
    decl.loadCdfs(options.cdfFile)
    # now, decl.cdfs is a dict, containing at each mass bin two np arrays, one containing the CDF values of the sigma_m_over_m of the reference dataset, and the other the corresponding x-values (0,...,0.5) 



    #Reading the background mgg files in one go with pandas
    import pandas as pd
    import glob


    files = glob.glob( str(options.infile) + "*.parquet")
    data = [pd.read_parquet(f) for f in files]
    #data = pd.read_parquet("/net/scratch_cms3a/daumann/massresdecorrhiggsdna/big_bkg/Diphoton.parquet")
    events= pd.concat(data,ignore_index=True)

    df = pandas.DataFrame()
    df["sigma_m_over_m"] = events.sigma_m_over_m_Smeared.to_numpy()
    df["mass"] = events.mass.to_numpy()
    df["weight"] = events.weight.to_numpy()

    print("var, dVar:", options.var, options.dVar)
    decl.df = df.loc[:, [options.var, options.dVar]]
    decl.df.reset_index(inplace=True)

    # options.ref is the mass bin (125.)
    df['{}_decorr'.format(options.var)] = decl.doDecorr(options.ref)
    print( options.ref, options.var , options.dVar )

    
    if 'sigmaMoM_decorr' in df.columns:
        df['sigmaMoM_decorrOld'] = df['sigmaMoM_decorr']
    
    if options.var == 'sigmarv':
        df['sigmaMoM_decorr'] = df['sigmarv_decorr']

    if options.var == 'sigmaRV':
        df['sigmaMoM_decorr'] = df['sigmaRV_decorr']

    print("INFO: df after decorr:\n", df.head(10))
    # df.to_root(options.outFile, key=tree, mode='a')

    events["sigma_m_over_m_decorr"] = decl.doDecorr(options.ref)

    #Writing the new 'tree' with the decorrelated 
    events.to_parquet(options.outFile)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-t','--tree', nargs='+', required=True)
    requiredArgs.add_argument('-i', '--infile', action='store', type=str, required=True)
    requiredArgs.add_argument('-c','--cdfFile', action='store', type=str, required=True)
    requiredArgs.add_argument('-v','--var', action='store', type=str, required=True)
    requiredArgs.add_argument('-d','--dVar', action='store', type=str, required=True)
    requiredArgs.add_argument('-o','--outFile', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-r', '--ref', action='store', type=float, default=125.)
    optArgs.add_argument('--columns', nargs='+')
    optArgs.add_argument('--nomColumns', nargs='+')
    optArgs.add_argument('--vecColumns', nargs='+')
    options = parser.parse_args()
    main(options)
