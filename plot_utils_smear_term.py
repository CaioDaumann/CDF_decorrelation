#this script objective is to make the studies of mass resolution with the addition of the SS smearing term
# I decided to do it, so the former ploter code can be focused only in mass decorrelation stuff!

import os 
import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
import boost_histogram as bh 
import matplotlib.pyplot as plt 
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import pandas as pd
import scipy

import mplhep as hep

import glob

def kinematical_rw( data, mc ,mc_weights, probe_data, probe_mc ):

        variable_names = [ "pt", "eta", "phi", "rho" ]

        mc_weights_before = mc_weights 
        #creating the mc and data weight histograms
        mc_weights = np.ones_like( mc[:,0] )
        mc_weights = mc_weights/np.sum( mc_weights )

        data_weights = np.ones_like( data[:,0] )
        data_weights = data_weights/np.sum( data_weights )

        #taking the log of the pt, in order to remove tails and make it easier to perform the rw
        #mc[:,0]   = np.log( mc[:,0] )
        #data[:,0] = np.log( data[:,0] )

        #pt_min,pt_max = np.min([ np.min( np.array(mc[:,0]) ),np.min( data[:,0] ) ]) ,np.max( [ np.max( np.array(mc[:,0]) ),np.max( data[:,0] ) ] )
        pt_min,pt_max   = 40, 160
        rho_min,rho_max = 7,60
        eta_min,eta_max = -2.501,2.501
        phi_min, phi_max = -3.15,3.15
        n_bins = 20
        pt_bins = 12
        #n_bins = 20
        eta_bins = 5
        phi_bins = 4

        #reweigthing only pt and rho for now:
        mc_histo,   edges = np.histogramdd( sample =  (mc[:,0] ,   mc[:,3],      mc[:,1]) ,  bins = (pt_bins,n_bins,eta_bins), range = [   [pt_min,pt_max], [ rho_min, rho_max ],[eta_min,eta_max] ], weights = mc_weights )
        data_histo, edges = np.histogramdd( sample =  (data[:,0] , data[:,3] , data[:,1]),  bins = (pt_bins,n_bins,eta_bins), range = [   [pt_min,pt_max], [ rho_min, rho_max ],[eta_min,eta_max] ], weights = data_weights )

        #we need to have a index [i,j] to each events, so we can rewighht based on data[i,j]/mc[i,j]

        pt_index =  np.array(pt_bins*( mc[:,0] -  pt_min) /(pt_max - pt_min)     , dtype=np.int8 )
        rho_index = np.array(n_bins*( mc[:,3] - rho_min )/(rho_max - rho_min) , dtype=np.int8  )
        eta_index = np.array(eta_bins*( mc[:,1] - eta_min )/(eta_max - eta_min) , dtype=np.int8  )
        phi_index = np.array(phi_bins*( mc[:,2] - phi_min )/(phi_max - phi_min) , dtype=np.int8  )

        #making sure we do not overflow the SF vector indices!
        #pt_index[pt_index > pt_bins -2 ] = pt_bins -2
        #pt_index[pt_index <= 0 ] = 0

        #rho_index[rho_index > n_bins -2 ] = n_bins -2
        #rho_index[rho_index <= 0 ] = 0

        #eta_index[eta_index > eta_bins -2 ] = eta_bins -2
        #eta_index[eta_index <= 0 ] = 0

        ###################
        #Now for the probe electron!
        probe_pt_min,probe_pt_max   = 22, 120
        probe_rho_min,probe_rho_max = 7,60
        probe_eta_min,probe_eta_max = -2.501,2.501
        phi_min, phi_max = -3.15,3.15
        n_bins = 20
        pt_bins = 12
        #n_bins = 20
        eta_bins = 5
        
        probe_mc_histo,   edges = np.histogramdd( sample =  (probe_mc[:,0] ,   probe_mc[:,3],      probe_mc[:,1]) ,  bins = (pt_bins,n_bins,eta_bins), range = [   [pt_min,pt_max], [ rho_min, rho_max ],[eta_min,eta_max] ], weights = mc_weights )
        probe_data_histo, edges = np.histogramdd( sample =  (probe_data[:,0] , probe_data[:,3] , probe_data[:,1]),  bins = (pt_bins,n_bins,eta_bins), range = [   [pt_min,pt_max], [ rho_min, rho_max ],[eta_min,eta_max] ], weights = data_weights )


        probe_pt_index =  np.array(pt_bins*( probe_mc[:,0] -  pt_min) /(pt_max - pt_min)     , dtype=np.int8 )
        probe_rho_index = np.array(n_bins*( probe_mc[:,3] - rho_min )/(rho_max - rho_min) , dtype=np.int8  )
        probe_eta_index = np.array(eta_bins*( probe_mc[:,1] - eta_min )/(eta_max - eta_min) , dtype=np.int8  )
        probe_phi_index = np.array(phi_bins*( probe_mc[:,2] - phi_min )/(phi_max - phi_min) , dtype=np.int8  )


        ###################

        #calculating the SF
        mc_weights = mc_weights* ( ( data_histo[  pt_index, rho_index,eta_index] )/(mc_histo[pt_index, rho_index,eta_index] + 1e-10 ) + 1.5*( probe_data_histo[  probe_pt_index, probe_rho_index,probe_eta_index] )/(probe_mc_histo[probe_pt_index, probe_rho_index,probe_eta_index] + 1e-10 ) )/2.

        #lets first plot it without rw
        
        for i in range( np.shape(data)[1] ):
            
            mean = np.mean(  np.nan_to_num(data[:,i]) )
            std  = np.std(   np.nan_to_num(data[:,i]) )

            data_hist     = hist.Hist(hist.axis.Regular(40, mean - 2*std, mean + 2*std))
            mc_hist       = hist.Hist(hist.axis.Regular(40, mean - 2*std, mean + 2*std))
            mc_rw_hist       = hist.Hist(hist.axis.Regular(40, mean - 2*std, mean + 2*std))
            
            data_hist.fill( np.array( data[:,i] ) )
            mc_hist.fill( np.array( mc[:,i] ) , weight = mc_weights_before  )
            mc_rw_hist.fill( np.array( mc[:,i] ) , weight = mc_weights  )
            
            ploter( mc_hist, data_hist , "plots_smear/rw_validation/" + str(variable_names[i]) + " .png" ,  xlabel = str(variable_names[i]), third_histo = mc_rw_hist )

        # Now, for the probe one!
        for i in range( np.shape(probe_data)[1] ):
            
            mean = np.mean(  np.nan_to_num(probe_data[:,i]) )
            std  = np.std(   np.nan_to_num(probe_data[:,i]) )

            data_hist     = hist.Hist(hist.axis.Regular(40, mean - 2*std, mean + 2*std))
            mc_hist       = hist.Hist(hist.axis.Regular(40, mean - 2*std, mean + 2*std))
            mc_rw_hist       = hist.Hist(hist.axis.Regular(40, mean - 2*std, mean + 2*std))
            
            data_hist.fill( np.array( probe_data[:,i] ) )
            mc_hist.fill( np.array( probe_mc[:,i] ) , weight = mc_weights_before  )
            mc_rw_hist.fill( np.array( probe_mc[:,i] ) , weight = mc_weights  )
            
            ploter( mc_hist, data_hist , "plots_smear/rw_validation/probe_" + str(variable_names[i]) + " .png" ,  xlabel = str(variable_names[i]), third_histo = mc_rw_hist )        



        return mc_weights

#main ploting function!
def ploter( mc_hist, data_hist, output_filename, xlabel = False, third_histo = False ):

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    hep.histplot(
        mc_hist,
        label = "Simulation",
        density = True,
        linewidth=3,
        ax=ax[0]
    )

    hep.histplot(
        data_hist,
        label = "Data (C+D)",
        yerr=True,
        density = True,
        color="black",
        histtype='errorbar',
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[0]
    )

    if( third_histo ):
        hep.histplot(
        third_histo,
        label = "Simulation Rw",
        density = True,
        color = "red",
        linewidth=3,
        ax=ax[0]
    )

    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.1*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    # line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)#, alpha=0.5)

    # Plot the ratios
    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy = mc_hist.to_numpy()

    integral_data = data_hist.sum() * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
    integral_mc = mc_hist.sum() * (mc_hist_numpy[1][1] - mc_hist_numpy[1][0])

    ratio = (data_hist_numpy[0] / integral_data) / (mc_hist_numpy[0] / integral_mc)

    ratio = np.nan_to_num(ratio)

    # ratio of the third histogram - reweighted one!
    if( third_histo ):

        #data_hist_numpy = data_hist.to_numpy()
        third_histo_numpy = third_histo.to_numpy()

        #integral_data = data_hist.sum() * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
        integral_mc = third_histo.sum() * (third_histo_numpy[1][1] - third_histo_numpy[1][0])

        ratio_third = (data_hist_numpy[0] / integral_data) / (third_histo_numpy[0] / integral_mc)

        ratio_third = np.nan_to_num(ratio_third)

    ### errors of numerator and denominator separately plotted:
    errors_nom = (np.sqrt(data_hist_numpy[0])/integral_data) / (mc_hist_numpy[0] / integral_mc)

    errors_den = np.sqrt(mc_hist_numpy[0]) / mc_hist_numpy[0]

    #print("errors_den", errors_den)

    errors_nom = np.abs(np.nan_to_num(errors_nom))
    errors_den = np.abs(np.nan_to_num(errors_den))

    lower_bound = 1 - errors_den
    upper_bound = 1 + errors_den

    # Plot the hatched region
    ax[1].fill_between(data_hist_numpy[1][:-1],
        lower_bound,
        upper_bound,
        hatch='XXX',
        alpha=0.9,
        facecolor="none",
        edgecolor="tab:blue", 
        linewidth=0
    )

    if( third_histo ):
        hep.histplot(
            ratio_third,
            bins=data_hist_numpy[1],
            label=None,
            color="red",
            histtype='errorbar',
            yerr=errors_nom,
            markersize=12,
            elinewidth=3,
            alpha=1,
            ax=ax[1]
        )

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="blue",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )



    ax[0].set_ylabel("Fraction of events / GeV", fontsize=26)
    ax[1].set_ylabel("Data / MC", fontsize=26)
    
    if( xlabel ):
        ax[1].set_xlabel( xlabel, fontsize=26)
    else:
        ax[1].set_xlabel("Diphoton mass [GeV]", fontsize=26)
    

    ax[0].tick_params(labelsize=24)
    ax[1].set_ylim(0., 1.1*ax[0].get_ylim()[1])
    ax[1].set_ylim(0.5, 1.5)

    ax[0].legend(
        loc="upper right", fontsize=20
    )

    hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=21.7)

    plt.subplots_adjust(hspace=0.03)

    plt.tight_layout()

    fig.savefig(output_filename)

    return 0

# Calculates the invariant mass of the electron pair!
def mass_dist(mass_inputs_data):

    mass_inputs_data = np.array(mass_inputs_data) 

    mass_data = np.sqrt(  2*mass_inputs_data[:,0]*mass_inputs_data[:,3]*( np.cosh(  mass_inputs_data[:,1] -  mass_inputs_data[:,4]  )  - np.cos( mass_inputs_data[:,2]  -mass_inputs_data[:,5] )  )  )

    return mass_data

def main():
    
    #lets read the MC and data files!
    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/DY_postEE_v12/nominal/*.parquet")
    # /net/scratch_cms3a/daumann/HiggsDNA/S_S_tests_w_pileuprw
    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests_w_pileuprw/DY_postEE_v12/nominal/*.parquet")
    #files = files[:20] #remove this - for tests only!
    
    # last test before dream
    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/zmmg_files/DY_preEE_v12/nominal/*.parquet")  
    files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/flow_tests_sigma_m/DY_postEE_v12/nominal/*.parquet")   
    

    data = [pd.read_parquet(f) for f in files]
    vector= pd.concat(data,ignore_index=True)

    mc_sigma_m_over_m            =     np.array(vector["sigma_m_over_m_corr"])
    mc_sigma_m_over_m_Smeared    =     np.array(vector["sigma_m_over_m_Smeared_corrected"])
    
    mc_tag_sigma_e                   =     np.array(vector["tag_energyErr"])
    mc_tag_sigma_e_smeared           =     np.array(vector["tag_energyErr_Smeared"])

    mc_probe_sigma_e                   =     np.array(vector["probe_energyErr"])
    mc_probe_sigma_e_smeared           =     np.array(vector["probe_energyErr_Smeared"])    
    
    mc_energy_tag                      =     np.array(vector["tag_pt"])*np.cosh( np.array(vector["tag_eta"]) )
    mc_energy_probe                    =     np.array(vector["probe_pt"])*np.cosh( np.array(vector["probe_eta"]) )

    mc_rho                             =     np.array( vector["fixedGridRhoAll"] )

    mc_weights                   =     np.array(vector["weight"])

    mc_kinematics                      =     np.concatenate( [ np.array(vector["tag_pt"]).reshape(-1,1), np.array(vector["tag_eta"]).reshape(-1,1) , np.array(vector["tag_phi"]).reshape(-1,1) ,np.array(vector["fixedGridRhoAll"]).reshape(-1,1)], axis = 1 )
    mc_kinematics_probe                =     np.concatenate( [ np.array(vector["probe_pt"]).reshape(-1,1), np.array(vector["probe_eta"]).reshape(-1,1) , np.array(vector["probe_phi"]).reshape(-1,1) ,np.array(vector["fixedGridRhoAll"]).reshape(-1,1)], axis = 1 )

    #since there is no mass in the ntuples 
    mc_mass = mass_dist( np.concatenate( [ np.array(vector["tag_pt"]).reshape(-1,1) , np.array(vector["tag_eta"]).reshape(-1,1) , np.array(vector["tag_phi"]).reshape(-1,1) , np.array(vector["probe_pt"]).reshape(-1,1) , np.array(vector["probe_eta"]).reshape(-1,1) , np.array(vector["probe_phi"]).reshape(-1,1)   ], axis = 1 ) )

    #mask to select only events inside the window of 80 and 100 GeV
    #lets also sincronize with florians cuts [tag pt > 40] - HLT_Ele32
    mask_mass =  np.logical_and( mc_mass >= 80, mc_mass <= 100)
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["tag_eta"]) ) < 2.5 )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["tag_pt"]) ) < 160 )
    mask_mass =  np.logical_and( mask_mass , np.array( vector["fixedGridRhoAll"] ) > 7 )
    mask_mass =  np.logical_and( mask_mass , np.array( vector["fixedGridRhoAll"] ) < 60 )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["probe_eta"]) ) < 2.5 )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["probe_pt"]) ) < 120 )

    #lets make the electron veto cut!
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["tag_electronVeto"]) ) == False )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["probe_electronVeto"]) ) == False )

    # also, the cutBased ID!
    mask_mass =  np.logical_and( mask_mass , np.array(vector["tag_cutBased"]) > 0 )
    mask_mass =  np.logical_and( mask_mass , np.array(vector["probe_cutBased"]) > 0 )
    
    
    mc_mass, mc_sigma_m_over_m,mc_sigma_m_over_m_Smeared,mc_weights= mc_mass[mask_mass],mc_sigma_m_over_m[mask_mass], mc_sigma_m_over_m_Smeared[mask_mass], mc_weights[mask_mass]

    mc_tag_sigma_e, mc_tag_sigma_e_smeared = mc_tag_sigma_e[mask_mass], mc_tag_sigma_e_smeared[mask_mass]
    mc_probe_sigma_e, mc_probe_sigma_e_smeared = mc_probe_sigma_e[mask_mass], mc_probe_sigma_e_smeared[mask_mass]
    mc_energy_tag, mc_energy_probe = mc_energy_tag[mask_mass], mc_energy_probe[mask_mass]
    mc_rho = mc_rho[mask_mass]
    mc_kinematics = mc_kinematics[mask_mass]
    mc_kinematics_probe = mc_kinematics_probe[mask_mass]

    #now, reading the data files
    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/Data_run2022G_v12/nominal/*.parquet")
    #files2 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/Data_run2022F_v12/nominal/*.parquet")
    #files3 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/Data_run2022E_v12/nominal/*.parquet")
    
    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests_w_pileuprw/Data_run2022G_v12/nominal/*.parquet")
    #files2 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests_w_pileuprw/Data_run2022F_v12/nominal/*.parquet")
    #files3 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests_w_pileuprw/Data_run2022E_v12/nominal/*.parquet")

    #files = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/zmmg_files/Data_Run2022D_v12/nominal/*.parquet")
    #files2 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/zmmg_files/Data_Run2022C_v12/nominal/*.parquet")
    #files3 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/zmmg_files/Data_run2022E_v12/nominal/*.parquet")

    # For the corrected samples!
    files = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/flow_tests_sigma_m/Data_run2022F_v12/nominal/*.parquet")
    files2 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/flow_tests_sigma_m/Data_Run2022G_v12/nominal/*.parquet")
    
    files = [files, files2]#, files3] #concatenating the F+G files!
    data = [pd.read_parquet(f) for f in files]
    vector= pd.concat(data,ignore_index=True)


    data_sigma_m_over_m                 =     np.array(vector["sigma_m_over_m"])
    data_sigma_m_over_m_Smeared         =     np.array(vector["sigma_m_over_m_Smeared"])
    
    data_tag_sigma_e                     =     np.array(vector["tag_energyErr"])
    data_tag_sigma_e_smeared             =     np.array(vector["tag_energyErr_Smeared"])
    
    data_probe_sigma_e                   =     np.array(vector["probe_energyErr"])
    data_probe_sigma_e_smeared           =     np.array(vector["probe_energyErr_Smeared"])

    data_energy_tag                      =     np.array(vector["tag_pt"])*np.cosh( np.array(vector["tag_eta"]) )
    data_energy_probe                    =     np.array(vector["probe_pt"])*np.cosh( np.array(vector["probe_eta"]) )

    data_rho                             =     np.array( vector["fixedGridRhoAll"] )

    data_kinematics                      =     np.concatenate( [ np.array(vector["tag_pt"]).reshape(-1,1), np.array(vector["tag_eta"]).reshape(-1,1) , np.array(vector["tag_phi"]).reshape(-1,1) , np.array(vector["fixedGridRhoAll"]).reshape(-1,1)] , axis = 1)
    data_kinematics_probe                =     np.concatenate( [ np.array(vector["probe_pt"]).reshape(-1,1), np.array(vector["probe_eta"]).reshape(-1,1) , np.array(vector["probe_phi"]).reshape(-1,1) , np.array(vector["fixedGridRhoAll"]).reshape(-1,1)] , axis = 1)

    data_weights            =        np.ones( len( data_sigma_m_over_m ) ) 
    data_mass = mass_dist( np.concatenate( [ np.array(vector["tag_pt"]).reshape(-1,1) , np.array(vector["tag_eta"]).reshape(-1,1), np.array(vector["tag_phi"]).reshape(-1,1)  , np.array(vector["probe_pt"]).reshape(-1,1) , np.array(vector["probe_eta"]).reshape(-1,1) , np.array(vector["probe_phi"]).reshape(-1,1)   ], axis = 1 ) )   


    #mask data interval 
    mask_mass =  np.logical_and( data_mass >= 80, data_mass <= 100)
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["tag_eta"]) ) < 2.5 )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["tag_eta"]) ) < 160 )
    mask_mass =  np.logical_and( mask_mass , np.array( vector["fixedGridRhoAll"] ) > 7 )
    mask_mass =  np.logical_and( mask_mass , np.array( vector["fixedGridRhoAll"] ) < 60 )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["probe_eta"]) ) < 2.5 )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["probe_pt"]) ) < 120 )    

    #lets make the electron veto cut!
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["tag_electronVeto"]) ) == False )
    mask_mass =  np.logical_and( mask_mass , np.abs( np.array(vector["probe_electronVeto"]) ) == False )

    # also, the cutBased ID!
    mask_mass =  np.logical_and( mask_mass , np.array(vector["tag_cutBased"]) > 0 )
    mask_mass =  np.logical_and( mask_mass , np.array(vector["probe_cutBased"]) > 0 )

    data_mass, data_sigma_m_over_m,data_sigma_m_over_m_Smeared,data_weights= data_mass[mask_mass],data_sigma_m_over_m[mask_mass], data_sigma_m_over_m_Smeared[mask_mass], data_weights[mask_mass]
    
    data_tag_sigma_e, data_tag_sigma_e_smeared = data_tag_sigma_e[mask_mass], data_tag_sigma_e_smeared[mask_mass]
    data_probe_sigma_e, data_probe_sigma_e_smeared = data_probe_sigma_e[mask_mass], data_probe_sigma_e_smeared[mask_mass]
    data_energy_tag, data_energy_probe = data_energy_tag[mask_mass], data_energy_probe[mask_mass]
    data_rho = data_rho[mask_mass]
    data_kinematics = data_kinematics[mask_mass]
    data_kinematics_probe = data_kinematics_probe[mask_mass]

    #after all the selections and masks, lets try to perform a rw on the kinematical variables!
    mc_weights_rw = kinematical_rw( data_kinematics, mc_kinematics ,mc_weights, data_kinematics_probe, mc_kinematics_probe )

    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 0.035))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.035))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.035))

    data_hist.fill( data_sigma_m_over_m , weight = data_weights)
    mc_hist.fill(   mc_sigma_m_over_m   , weight = mc_weights)
    mc_rw_hist.fill(   mc_sigma_m_over_m   , weight = mc_weights_rw)

    ploter( mc_hist, data_hist, "plots_smear/nominal_sigmam_overm.png",  xlabel = r'$\sigma_{M}/M$ ' )#, third_histo = mc_rw_hist )

    #now plotting the smeared term!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 0.035))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.035))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.035))

    #data_hist.fill( data_sigma_m_over_m_Smeared , weight = data_weights)
    data_hist.fill( data_sigma_m_over_m_Smeared , weight = data_weights)
    mc_hist.fill(   mc_sigma_m_over_m_Smeared   , weight = mc_weights)
    mc_rw_hist.fill(   mc_sigma_m_over_m_Smeared   , weight = mc_weights_rw)
    

    ploter( mc_hist, data_hist, "plots_smear/smeared_sigmam_overm.png" ,  xlabel = r'$\sigma_{M}^{Smeared}/M$ ')#, third_histo = mc_rw_hist )

    #########
    #### Now just sigmaE/E
    ###########

    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    

    data_hist.fill( data_tag_sigma_e/data_energy_tag , weight = data_weights)
    mc_hist.fill(   mc_tag_sigma_e/mc_energy_tag   , weight = mc_weights)
    mc_rw_hist.fill(   mc_tag_sigma_e/mc_energy_tag   , weight = mc_weights_rw)

    ploter( mc_hist, data_hist, "plots_smear/nominal_tag_sigmae_overe.png" ,  xlabel = r'tag $\sigma_{E}/E$ ')#, third_histo = mc_rw_hist )

    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))

    data_hist.fill( data_probe_sigma_e/data_energy_probe , weight = data_weights)
    mc_hist.fill(   mc_probe_sigma_e/mc_energy_probe   , weight = mc_weights)
    mc_rw_hist.fill(   mc_probe_sigma_e/mc_energy_probe   , weight = mc_weights_rw)

    ploter( mc_hist, data_hist, "plots_smear/nominal_probe_sigmae_overe.png" ,  xlabel = r'probe $\sigma_{E}/E$ ')#, third_histo = mc_rw_hist )

    #### SMEARED ONES BELOW - ALREADY DIVIDED BY E  

    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))

    data_hist.fill( data_tag_sigma_e_smeared , weight = data_weights)
    mc_hist.fill(   mc_tag_sigma_e_smeared   , weight = mc_weights)
    mc_rw_hist.fill(   mc_tag_sigma_e_smeared   , weight = mc_weights_rw)

    ploter( mc_hist, data_hist, "plots_smear/smeared_tag_sigmae_overe.png",  xlabel = r'tag $\sigma_{E}^{smeared}/E$ ' )#, third_histo = mc_rw_hist )

    #now for the probe electron!

    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 0.06))

    data_hist.fill( data_probe_sigma_e_smeared , weight = data_weights)
    mc_hist.fill(   mc_probe_sigma_e_smeared   , weight = mc_weights)
    mc_hist.fill(   mc_probe_sigma_e_smeared   , weight = mc_weights_rw)

    ploter( mc_hist, data_hist, "plots_smear/smeared_probe_sigmae_overe.png" , xlabel = r'probe $\sigma_{E}^{smeared}/E$ ')#, third_histo = mc_rw_hist )    

    #lets plot the energy distributions now:
    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 100))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 100))
    mc_rw_hist       = hist.Hist(hist.axis.Regular(80, 0, 100))

    data_hist.fill( data_energy_probe , weight = data_weights)
    mc_hist.fill(   mc_energy_probe   , weight = mc_weights)
    mc_rw_hist.fill(   mc_energy_probe   , weight = mc_weights_rw)

    ploter( mc_hist, data_hist, "plots_smear/energy_probe.png" , xlabel = r'probe Energy [GeV]')#, third_histo = mc_rw_hist )  

    #lets plot the energy distributions now:
    #lets use a plotter with pull plots!
    data_hist     = hist.Hist(hist.axis.Regular(80, 0, 100))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 0, 100))

    data_hist.fill( data_energy_tag , weight = data_weights)
    mc_hist.fill(   mc_energy_tag   , weight = mc_weights)

    ploter( mc_hist, data_hist, "plots_smear/energy_tag.png" , xlabel = r'tag Energy [GeV]' )  


    #now, pileup information!!!

    data_hist     = hist.Hist(hist.axis.Regular(80, 10, 50))
    mc_hist       = hist.Hist(hist.axis.Regular(80, 10, 50))

    data_hist.fill( data_rho , weight = data_weights)
    mc_hist.fill(   mc_rho   , weight = mc_weights)

    ploter( mc_hist, data_hist, "plots_smear/event_pileup.png" , xlabel = r'event rho' )  

if __name__ == "__main__":
    main()