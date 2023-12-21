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

def ploting_sigma_distributions( sigma_m_over_m_before_smear, sigma_m_over_m_smeared , sigma_m_over_m_decorr, mc_weights ):

    bins = np.linspace(0.0,0.03,80 )
    plt.hist( sigma_m_over_m_smeared, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'orange' , label= r'after smearing' )
    plt.hist( sigma_m_over_m_before_smear, weights=mc_weights,bins = bins ,linestyle='dashed', histtype=u'step',linewidth = 3, color = 'blue', label= r'before smearing' )
    #plt.hist( sigma_term, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'blue', label= 'smearing term' )
    plt.legend()
    plt.xlabel( r'$\sigma_{m}/m$' )
    plt.ylabel('density')

    plt.savefig( 'plots/histograms.png' )
    
    plt.close()
    
    plt.hist( sigma_m_over_m_decorr, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'orange' , label= 'decorrelated' )
    plt.hist( sigma_m_over_m_smeared, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'blue',   label= 'before decorrelation' )
    
    plt.legend()
    plt.xlabel( r'$\sigma_{m}/m$' )
    plt.ylabel('density')
    plt.savefig( 'plots/smear_histograms.png' )

def plot_mass_dist( mc_hist, data_hist, output_filename,region=None ):

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    hep.histplot(
        mc_hist,
        label = "Correlated",
        yerr=True,
        density = True,
        linewidth=3,
        ax=ax[0]
    )

    hep.histplot(
        data_hist,
        label = "Decorrelated",
        yerr=True,
        density = True,
        color="black",
        histtype='errorbar',
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[0]
    )

    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.05*ax[0].get_ylim()[1])
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

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="black",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )

    ax[0].set_ylabel("Fraction of events / GeV", fontsize=26)
    ax[1].set_ylabel("Data / MC", fontsize=26)
    ax[1].set_xlabel("Diphoton mass [GeV]", fontsize=26)
    if region:
        if not "ZpT" in region:
            ax[0].text(0.05, 0.75, "Region: " + region.replace("_", "-"), fontsize=22, transform=ax[0].transAxes)
        else:
            ax[0].text(0.05, 0.75, "Region: " + region.split("_ZpT_")[0].replace("_", "-"), fontsize=22, transform=ax[0].transAxes)
            ax[0].text(0.05, 0.68, r"$p_\mathrm{T}(Z)$: " + region.split("_ZpT_")[1].replace("_", "-") + "$\,$GeV", fontsize=22, transform=ax[0].transAxes)
    ax[0].tick_params(labelsize=24)
    ax[1].set_ylim(0., 1.1*ax[0].get_ylim()[1])
    ax[1].set_ylim(0.5, 1.5)

    ax[0].legend(
        loc="upper right", fontsize=24
    )

    hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=21.7)

    plt.subplots_adjust(hspace=0.03)

    plt.tight_layout()

    fig.savefig(output_filename)

    return 0

def plot_inclusive_CDF( correlated, decorrelated, mc_weights ):

        val = correlated
        dBins = dBins=np.linspace(0.,0.5,1001)
        hist, _ = np.histogram(val, weights=mc_weights, bins=dBins)
        rightEdge = dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum /= bCum.max()
        cdfBinned = np.vstack((bCum,rightEdge))

        val = decorrelated
        dBins = dBins=np.linspace(0.,0.5,1001)
        hist, _ = np.histogram(val, weights=mc_weights, bins=dBins)
        rightEdge = dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum /= bCum.max()
        cdfBinned_deco = np.vstack((bCum,rightEdge))

        #Ploting:
        plt.close()
        plt.plot( cdfBinned[1], cdfBinned[0], color='blue'  , alpha=0.7, linewidth = 2,label = ' Correlated')
        plt.plot( cdfBinned_deco[1], cdfBinned_deco[0], color='red'   , alpha=0.7, linewidth = 2,label = ' Decorrelated ')
        plt.legend()
        plt.ylim( 0,1.1 )
        plt.xlim(0,0.05)
        plt.ylabel( 'CDF' )
        plt.xlabel(r'$\sigma_{m}/m$')
        plt.savefig( 'plots/Inclusive_CDF.png' )
        plt.close()


def weighted_median_(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

#def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
#    i = np.argsort(values)
#    c = np.cumsum(weights[i])
#    q = np.searchsorted(c, quantiles * c[-1])
#    return np.where(c[q]/c[-1] == quantiles, 0.5 * (values[i[q]] + values[i[q+1]]), values[i[q]])

def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

def calculate_bins_position(mass, weights):

    #This code below is responsable for separate the mass events in 8 bins of equal number of events
    sorted_data = np.sort(mass) #This sorts the mass in increasing order

    # Calculate the number of elements in each bin
    num_elements = len(sorted_data) // 40

    # Initialize a list to store the bins
    bins,x_bins = [],[]
    x_bins.append(100)

    # Create 8 bins with the same number of events
    for i in range(0, len(sorted_data), num_elements):
        bin_data = sorted_data[i:i + num_elements]
        bins.append(bin_data)
        x_bins.append( float(np.max(bin_data)) )

    x_bins.pop()
    x_bins.pop()
    x_bins.append(180)

    return x_bins

def plot_sigma_over_m_profile(position, mean_value, mean_value_decorr,mean_value_10,mean_value_decorr_10,mean_value_90,mean_value_decorr_90):
    
    plt.close()

    #10% quantile
    plt.plot( position, mean_value_decorr_10 ,linestyle='dashed', linewidth = 4, color = 'orange' )
    plt.plot( position, mean_value_10        ,linestyle='dashed', linewidth = 4, color = 'blue'   )

    plt.plot([], [], ' ', label="Diphoton samples")

    #50 quantile
    plt.plot( position, mean_value_decorr , linewidth = 4, color = 'orange', label =   "decorrelated")
    plt.plot( position, mean_value , linewidth = 4,        color = 'blue'    ,label =  "not decorrelated")

    #90% quantile
    plt.plot( position, mean_value_decorr_90 ,linestyle='dashed', linewidth = 4, color = 'orange' )
    plt.plot( position, mean_value_90        ,linestyle='dashed', linewidth = 4, color = 'blue'   )

    x = [120,130]
    y = [0.0104,0.0104]
    #plt.plot(x, y, linewidth = 3, linestyle='--'color = 'red')
    plt.ylim( 0.0085, 0.0255 )
    plt.xlim( 95, 185 )
    plt.xlabel( 'Diphoton Mass [GeV]' )
    plt.ylabel( r'$\sigma_{m}/m$' )
    
    #plt.text(0.5, 0.75, "Diphoton MC samples", fontsize=22)
    
    hep.cms.label(data=True, loc=0, label="Work in progress", com=13.6, lumi=21.7)
    plt.tight_layout()
    plt.legend( fontsize = 16 ,loc = 'upper right', facecolor='white',  borderaxespad=1. ).set_zorder(2)
    plt.savefig('plots/sucess_decorr.png')
    plt.savefig('plots/sucess_decorr.pdf')

def plot_CDFS(mass, sigma_over_m,sigma_over_m_decorr,mc_weights):
    

    #First lets separate the mass and the uncertantities in three bins (100:100,5 ; 125:125,5 : )
    mass_mask_1 =  np.logical_and( mass > 100 , mass < 100.5 ) 
    mass_mask_2 =  np.logical_and( mass > 125 , mass < 125.5 ) 
    mass_mask_3 =  np.logical_and( mass > 170 , mass < 170.5 )

    great_mask = [mass_mask_1, mass_mask_2, mass_mask_3]
    great_CDF  = []

    #First, the plot of the sigma_over_m correlated CDF in the three bins
    for i in range( 3 ):
        
        val = sigma_over_m[ great_mask[i] ]

        dBins = dBins=np.linspace(0.,0.5,1001)

        hist, _ = np.histogram(val, weights=mc_weights[ great_mask[i] ], bins=dBins)
        rightEdge = dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum = bCum/float(bCum.max())
        cdfBinned = np.vstack((bCum,rightEdge))

        great_CDF.append(cdfBinned)

    #Ploting:
    plt.close()
    plt.plot( great_CDF[0][1], great_CDF[0][0], color='blue'  , alpha=0.7, linewidth = 2,label = r' 100 < $m_{\gamma\gamma}$ < 100.5 ')
    plt.plot( great_CDF[1][1], great_CDF[1][0], color='red'   , alpha=0.7, linewidth = 2,label = r' 125 < $m_{\gamma\gamma}$ < 125.5 ')
    plt.plot( great_CDF[2][1], great_CDF[2][0], color='purple', alpha=0.7, linewidth = 2,label = r' 170 < $m_{\gamma\gamma}$ < 170.5 ')
    
    hep.cms.label(data=True, loc=0, label="Work in progress", com=13.6, lumi=21.7)


    plt.legend(fontsize=22,loc = 'lower right')
    plt.xlim(0,0.035)
    plt.ylabel( r'$cdf(\sigma_{m}/m)$' )
    plt.xlabel(  r'$\sigma_{m}/m$')
    plt.savefig( 'plots/r_CDF.png' )
    plt.savefig( 'plots/r_CDF.pdf' )

    #Now, the same plot but using the decorrelated sigma_over_m

    great_CDF  = []

    #First, the plot of the sigma_over_m correlated CDF in the three bins
    for i in range( 3 ):
        
        val = sigma_over_m_decorr[ great_mask[i] ]

        dBins = dBins=np.linspace(0.,0.5,1001)

        hist, _ = np.histogram(val, weights=mc_weights[ great_mask[i] ], bins=dBins)
        rightEdge = dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum = bCum/float(bCum.max())
        cdfBinned = np.vstack((bCum,rightEdge))

        great_CDF.append(cdfBinned)

    #Ploting:
    plt.close()
    plt.plot( great_CDF[0][1], great_CDF[0][0], color='blue'  , alpha=0.7, linewidth = 2,label =  r' 100 < $m_{\gamma\gamma}$ < 100.5 ')
    plt.plot( great_CDF[1][1], great_CDF[1][0], color='red'   , alpha=0.7, linewidth = 2,label =  r' 125 < $m_{\gamma\gamma}$ < 125.5 ')
    plt.plot( great_CDF[2][1], great_CDF[2][0], color='purple', alpha=0.7, linewidth = 2,label =  r' 170 < $m_{\gamma\gamma}$ < 170.5 ')
    plt.legend(fontsize=22,loc = 'lower right')

    hep.cms.label(data=True, loc=0, label="Work in progress", com=13.6, lumi=21.7)


    plt.xlim(0,0.035)
    plt.ylabel( r'$cdf(\sigma_{m}^{Decorr}/m)$' )
    plt.xlabel( r'$\sigma_{m}^{Decorr}/m$')
    plt.savefig( 'plots/r_CDF_decorr.png' )
    plt.savefig( 'plots/r_CDF_decorr.pdf' )
    plt.close()

    #plot the transformatiom
    mass_mask_1 =  np.logical_and( mass > 100 , mass < 100.5 )
    sig_m        = sigma_over_m[mass_mask_1]
    sig_m_decorr = sigma_over_m_decorr[mass_mask_1]

    plt.scatter( sig_m,sig_m_decorr, label = ' 100 - 100.5 ')
    plt.xlim(0.004,0.025)
    plt.ylim(0.004,0.025)

       #x == y line
    lims = [
    np.min([0.004, 0.025]),  # min of both axes
    np.max([0.004, 0.025]),  # max of both axes
    ]

    # now plot both limits against eachother
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel( 'Input Value' )
    plt.ylabel( 'Corrected Value' )
    plt.legend()
    plt.tight_layout()
    plt.savefig( 'plots/transformation.png'  ) 
    plt.close()

    #plot the transformatiom - on signal range xD
    mass_mask_1 =  np.logical_and( mass > 160 , mass < 160.5 )
    sig_m        = sigma_over_m[mass_mask_1]
    sig_m_decorr = sigma_over_m_decorr[mass_mask_1]

    plt.scatter( sig_m,sig_m_decorr, color = 'orange', label = ' 160 - 160.5 ')
    plt.xlim(0.004,0.025)
    plt.ylim(0.004,0.025)

    #x == y line
    lims = [
    np.min([0.004, 0.025]),  # min of both axes
    np.max([0.004, 0.025]),  # max of both axes
    ]

    # now plot both limits against eachother
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)


    plt.xlabel( 'Input Value' )
    plt.ylabel( 'Corrected Value' )
    plt.legend()
    plt.tight_layout()
    plt.savefig( 'plots/transformation_signal.png'  ) 

    plt.close()
    #plot the transformatiom - on signal range xD
    mass_mask_1 =  np.logical_and( mass > 125 , mass < 125.5 )
    sig_m        = sigma_over_m[mass_mask_1]
    sig_m_decorr = sigma_over_m_decorr[mass_mask_1]

    plt.scatter( sig_m,sig_m_decorr, label = ' 125 - 125.5 ', color = 'red')
    plt.xlim(0.004,0.025)
    plt.ylim(0.004,0.025)

    #x == y line
    lims = [
    np.min([0.004, 0.025]),  # min of both axes
    np.max([0.004, 0.025]),  # max of both axes
    ]

    # now plot both limits against eachother
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)


    plt.xlabel( 'Input Value' )
    plt.ylabel( 'Corrected Value' )
    plt.legend()
    plt.tight_layout()
    plt.savefig( 'plots/transformation_signal_id.png'  ) 


    return None

def mass_distributions_plots(mass, sigma_m_over_m, sigma_m_over_m_decorr, mc_weights):
    
    #ploting the full mass distribution

    #Filling the histograms and performing masking and soritng operations
    mass_hist          = hist.Hist(hist.axis.Regular(60, 100, 180))
    mass_hist_quantile = hist.Hist(hist.axis.Regular(60, 100, 180))
    
    #Pllting the sigma_m_over_m decorrelated
    ind = np.argsort(sigma_m_over_m_decorr)
    mass_decorr, weight_decorr = mass[ind],mc_weights[ind]
    mass_hist.fill( mass_decorr[ :int( len( mass )*0.50 ) ], weight = weight_decorr[ :int( len( mass )*0.50 ) ] )
    
    ind = np.argsort(sigma_m_over_m)
    mass_quantile,weight_quantile = mass[ind], mc_weights[ind]

    mass_hist_quantile.fill( mass_quantile[ :int( len(mass)*0.50 ) ], weight = weight_quantile[ :int( len(mass)*0.50 ) ] )

    plot_mass_dist( mass_hist, mass_hist_quantile, 'plots/full_mass_distributions.png' )

    #Only first ten perceent quantile ####################################################################################
    mass_hist          = hist.Hist(hist.axis.Regular(60, 100, 180))
    mass_hist_quantile = hist.Hist(hist.axis.Regular(60, 100, 180))
    
    #Pllting the sigma_m_over_m decorrelated
    ind = np.argsort(sigma_m_over_m_decorr)
    mass_decorr, weight_decorr = mass[ind],mc_weights[ind]
    mass_hist.fill( mass_decorr[ :int( len( mass )*0.20 ) ], weight = weight_decorr[ :int( len( mass )*0.20 ) ] )
    
    ind = np.argsort(sigma_m_over_m)
    mass_quantile,weight_quantile = mass[ind], mc_weights[ind]

    mass_hist_quantile.fill( mass_quantile[ :int( len(mass)*0.20 ) ], weight = weight_quantile[ :int( len(mass)*0.20 ) ] )

    plot_mass_dist( mass_hist, mass_hist_quantile, 'plots/first_20_mass_distributions.png' )

    ######################################### Last 20 percent quantile ###################################################
    mass_hist          = hist.Hist(hist.axis.Regular(60, 100, 180))
    mass_hist_quantile = hist.Hist(hist.axis.Regular(60, 100, 180))
    
    #Pllting the sigma_m_over_m decorrelated
    ind = np.argsort(sigma_m_over_m_decorr)
    mass_decorr, weight_decorr = mass[ind],mc_weights[ind]
    mass_hist.fill( mass_decorr[ -int( len( mass )*0.20 ): ], weight = weight_decorr[ -int( len( mass )*0.20 ): ] )
    
    ind = np.argsort(sigma_m_over_m)
    mass_quantile,weight_quantile = mass[ind], mc_weights[ind]

    mass_hist_quantile.fill( mass_quantile[ -int( len(mass)*0.20 ): ], weight = weight_quantile[ -int( len(mass)*0.20 ): ] )

    plot_mass_dist( mass_hist, mass_hist_quantile, 'plots/last_20_mass_distributions.png' )


    return 0

    
import glob


def mass_dist(mass_inputs_data):

    mass_inputs_data = np.array(mass_inputs_data) 
    #mass_inputs_mc   = np.array(mass_inputs_mc)

    mass_data = np.sqrt(  2*mass_inputs_data[:,0]*mass_inputs_data[:,3]*( np.cosh(  mass_inputs_data[:,1] -  mass_inputs_data[:,4]  )  - np.cos( mass_inputs_data[:,2]  -mass_inputs_data[:,5] )  )  )
    #mass_  = np.sqrt(  2*mass_inputs_mc[:,0]*mass_inputs_mc[:,3]*( np.cosh(  mass_inputs_mc[:,1] -  mass_inputs_mc[:,4]  )  - np.cos( mass_inputs_mc[:,2]  -mass_inputs_mc[:,5] )  )  )

    return mass_data

def plot_data_and_mc_mass_ressolution():

    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/DY_postEE_v12/nominal/*.parquet")
    # H->yy files
    files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/tests_mr_excludelater/GluGluHToGG_postEE_M125_2022/nominal/*.parquet" )
    data = [pd.read_parquet(f) for f in files]
    vector= pd.concat(data,ignore_index=True)

    mc_sigma_m_over_m        =     np.array(vector["sigma_m_over_m"])
    mc_sigma_m_over_m_Smeared        =     np.array(vector["sigma_m_over_m_Smeared"])
    #mc_mass                  =     np.array(vector["mass"])
    mc_sigma_m_over_m_decorr =     np.array(vector["sigma_m_over_m_Smeared"])
    mc_weights            =     np.array(vector["weight"])

    mc_mass = mass_dist( np.concatenate( [ np.array(vector["lead_pt"]).reshape(-1,1) , np.array(vector["lead_eta"]).reshape(-1,1) , np.array(vector["lead_phi"]).reshape(-1,1) , np.array(vector["sublead_pt"]).reshape(-1,1) , np.array(vector["sublead_eta"]).reshape(-1,1) , np.array(vector["sublead_phi"]).reshape(-1,1)   ], axis = 1 ) )
    
    # Restricting eveything to the mass range of 100 and 180 - for diphoton samples
    mask_mass =  np.logical_and( mc_mass >= 100, mc_mass <= 180)
    #mask_mass =  np.logical_and( mask_mass , np.array(vector["tag_mvaID"]) > 0.0 )
    #mask_mass =  np.logical_and( mask_mass , np.array(vector["sublead_mvaID"]) > 0.6 )
    mc_mass, mc_sigma_m_over_m,mc_sigma_m_over_m_decorr,mc_weights= mc_mass[mask_mass],mc_sigma_m_over_m[mask_mass], mc_sigma_m_over_m_decorr[mask_mass], mc_weights[mask_mass]


    #files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/Data_run2022F_v12/nominal/*.parquet")
    #files2 = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/S_S_tests/Data_run2022F_v12/nominal/*.parquet")
    # diphoton samples
    files = glob.glob("/net/scratch_cms3a/daumann/HiggsDNA/tests_mr_excludelater/Diphoton_2022_postEE/nominal/*.parquet")
    #files = [files, files2]
    data = [pd.read_parquet(f) for f in files]
    vector= pd.concat(data,ignore_index=True)

    data_sigma_m_over_m        =     np.array(vector["sigma_m_over_m"])
    #data_sigma_m_over_m        =     np.array(vector["sigma_m_over_m_Smeared"])
    #data_mass                  =     np.array(vector["mass"])
    data_sigma_m_over_m_decorr =     np.array(vector["sigma_m_over_m_Smeared"])
    data_weights            =        np.ones( len( data_sigma_m_over_m ) ) #np.array(vector["weight"])

    data_mass = mass_dist( np.concatenate( [ np.array(vector["lead_pt"]).reshape(-1,1) , np.array(vector["lead_eta"]).reshape(-1,1) , np.array(vector["lead_phi"]).reshape(-1,1) , np.array(vector["sublead_pt"]).reshape(-1,1) , np.array(vector["sublead_eta"]).reshape(-1,1) , np.array(vector["sublead_phi"]).reshape(-1,1)   ], axis = 1 ) )

    mask_mass = 0
    mask_mass =  np.logical_and( data_mass >= 100, data_mass <= 180)
    #mask_mass =  np.logical_and( mask_mass , np.array(vector["tag_mvaID"]) > 0.0 )
    #mask_mass =  np.logical_and( mask_mass , np.array(vector["sublead_mvaID"]) > 0.6 )

    data_mass, data_sigma_m_over_m,data_sigma_m_over_m_decorr,data_weights= data_mass[mask_mass],data_sigma_m_over_m[mask_mass], data_sigma_m_over_m_decorr[mask_mass], data_weights[mask_mass]

    bins = np.linspace(0.0,0.04,120 )
    plt.hist( mc_sigma_m_over_m, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'orange' , label=r'$H\rightarrow \gamma\gamma$', density=True )
    plt.hist( data_sigma_m_over_m, weights=data_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'blue', label= 'Diphoton', density=True )
    #plt.hist( sigma_term, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'blue', label= 'smearing term' )
    plt.legend()
    plt.xlabel( r'$\sigma_{m}/m$' )
    plt.ylabel('Density')

    plt.savefig( 'plots/mc_data_mass_resolution.png' )
    plt.close()


    #now lets plot the MC and Data smeared distributuions

    bins = np.linspace(0.0,0.04,120 )
    plt.hist( mc_sigma_m_over_m_decorr, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'orange' , label=r'$H\rightarrow \gamma\gamma$', density=True )
    plt.hist( data_sigma_m_over_m_decorr, weights=data_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'blue', label= 'Diphoton', density=True )
    #plt.hist( sigma_term, weights=mc_weights,bins = bins , histtype=u'step',linewidth = 3, color = 'blue', label= 'smearing term' )
    plt.legend()
    plt.xlabel( r'$\sigma_{m}^{Smeared}/m$' )
    plt.ylabel('Density')

    plt.savefig( 'plots/mc_data_mass_resolution_smeared.png' )
    plt.close()



#ploting algorithm to plot the results of mass decorrelator algorithm
def main():

    #comparing the mass resolution distirbutions of data and Diphoton samples 
    plot_data_and_mc_mass_ressolution()
    exit()

    #Reading the .parquet file contatinng the original trees + the decorrelated mass resolution
    vector = ak.from_parquet("Diphoton_DNA_1.parquet")

    #files = glob.glob( "/net/scratch_cms3a/daumann/massresdecorrhiggsdna/data_files/DataF_2022/nominal/*.parquet")
    #files2 = glob.glob( "/net/scratch_cms3a/daumann/massresdecorrhiggsdna/data_files/DataG_2022/nominal/*.parquet")
    #files = [files, files2]
    #files = glob.glob( "/net/scratch_cms3a/daumann/massresdecorrhiggsdna/data_files/Diphoton/nominal/*.parquet")
    
    files = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/tests_mr_excludelater/Diphoton_2022_postEE/nominal/*.parquet" )
    data = [pd.read_parquet(f) for f in files]
    vector= pd.concat(data,ignore_index=True)

    #Reading the colums from the ak object -  sometimes you have to remove the np.array thing from here!
    sigma_m_over_m_before_smear    =     np.array(vector["sigma_m_over_m"])
    sigma_m_over_m                 =     np.array(vector["sigma_m_over_m_Smeared"])
    sigma_m_over_m_decorr          =     np.array(vector["sigma_m_over_m_decorr"])
    mass                           =     np.array(vector["mass"])
    sigma_term                     =     np.array(vector["sigma_m_over_m_decorr"])
    mc_weights                     =     np.array(vector["weight"])


    """
    #Filling some histogram for the distributions
    mass_h = hist.Hist(hist.axis.Regular(60, 100, 180))
    mass_h_again = hist.Hist(hist.axis.Regular(60, 100, 180))

    ind = np.argsort(sigma_m_over_m)
    mass_correlated,sigma_m_over_m = mass[ind], sigma_m_over_m[ind]


    mass_h.fill( mass_correlated[ :int( len(mass)*0.3 ) ] , weight= mc_weights[ :int( len(mass)*0.3 ) ] )
    
    #Lets choose only the events with the 20% best mass resolution here
    ind = np.argsort(sigma_m_over_m_decorr)
    mass_h_again.fill( mass, weight = mc_weights  )

    plot_mass_dist( mass_h, mass_h_again, 'plots/mass_distributions.png' )
    """
    
    #Performing some cuts on the mass, to stay in the 100 < m < 180 analysis range
    mask_mass =  np.logical_and( mass >= 100, mass <= 180)
    mask_mass =  np.logical_and( mask_mass, ~np.isnan(mass)  )
    #mask_mass =  np.logical_and( mask_mass , np.array(vector["lead_mvaID"])    > -1.6 )
    #mask_mass =  np.logical_and( mask_mass , np.array(vector["sublead_mvaID"]) > -1.6 )
    mass, sigma_m_over_m,sigma_m_over_m_decorr,mc_weights, sigma_m_over_m_before_smear= mass[mask_mass],sigma_m_over_m[mask_mass], sigma_m_over_m_decorr[mask_mass], mc_weights[mask_mass], sigma_m_over_m_before_smear[mask_mass]

    ploting_sigma_distributions( sigma_m_over_m_before_smear,sigma_m_over_m, sigma_m_over_m_decorr ,mc_weights )

    # Now, begin the profile plot calculations!
    x_bins = calculate_bins_position(mass,vector["weight"]) #This gives the mass values of separating the mass distirbution in 8 bins of equal number of events

    #This loop down here calculate the median of the sigma_over_m for the mass events in each of the 8 bins calculated above
    position, mean_value,mean_value_decorr,weighted_mean_value_decorr = [],[],[],[]

    mean_value_10,mean_value_decorr_10 = [],[]
    mean_value_90,mean_value_decorr_90 = [],[]
    for i in range( len(x_bins) - 1 ):
        #mass_min,mass_max = np.min( x_bins[i] ), np.max( x_bins[i+1] )
        mass_window = np.logical_and(  mass >=  x_bins[i], mass <=  x_bins[i+1])
        sigma_m_over_m_decorr_inside_window   = sigma_m_over_m_decorr[mass_window]
        sigma_m_over_m_inside_window          = sigma_m_over_m[mass_window]
        w_inside_window                       = mc_weights[mass_window]

        if( i == len(x_bins) - 2  ):
            position.append( 180 )
        elif( i == 0 ):
            position.append( 100 )
        else:
            position.append(   float( (x_bins[i] + x_bins[i+1])/2 ))
        mean_value.append(  weighted_quantiles_interpolate(sigma_m_over_m_inside_window,w_inside_window) )
        mean_value_decorr.append( weighted_quantiles_interpolate(sigma_m_over_m_decorr_inside_window,w_inside_window) )

        mean_value_10.append(  weighted_quantiles_interpolate(sigma_m_over_m_inside_window,w_inside_window,quantiles=0.10) )
        mean_value_decorr_10.append( weighted_quantiles_interpolate(sigma_m_over_m_decorr_inside_window,w_inside_window,quantiles=0.10) )

        mean_value_90.append(  weighted_quantiles_interpolate(sigma_m_over_m_inside_window,w_inside_window,quantiles=0.9) )
        mean_value_decorr_90.append( weighted_quantiles_interpolate(sigma_m_over_m_decorr_inside_window,w_inside_window,quantiles=0.9) )

    print( 'Checkpoint 2! ' )

    plot_sigma_over_m_profile(position, mean_value, mean_value_decorr,mean_value_10,mean_value_decorr_10,mean_value_90,mean_value_decorr_90)
    plot_CDFS(mass, sigma_m_over_m,sigma_m_over_m_decorr, mc_weights)

    #plot the transformation
    #making the plots after the cuts
    #plot_inclusive_CDF( sigma_m_over_m, sigma_m_over_m_decorr, mc_weights )
    mass_distributions_plots( mass,sigma_m_over_m,sigma_m_over_m_decorr, mc_weights )


    print( 'Acabou porra!' )

if __name__ == "__main__":
    main()

