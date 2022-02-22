import numpy  as np
import pandas as pd

import scipy.constants as constants
import scipy.stats     as stats
#import scipy.optimize  as optimize

import core.utils as ut
import core.efit  as efit


import core.pltext as pltext
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


ssamples = [r'$\beta\beta0\nu$', r'$^{214}$Bi', r'$^{208}$Tl']

erange     = (2.400, 2.650)
eroi       = (2.440, 2.475)
keys       = ['E', 'num_tracks', 'blob2_E', 'RoI']
varnames   = ['E', 'num_tracks', 'blob2_E', 'E']
varranges  = [erange, (1., 1.1), (0.4, np.inf), eroi]

#label   = 'track0_E'
erange   = (2.4, 2.7)
eroi     = (2.43,  2.48)
eblob2   = 0.4
#bins    = 100

abundace = 0.9
Qbb      = 2458 # keV Qbb value
W        = 135.9


def half_life(nbb, exposure, eff, abundance = 0.9, W = 135.9):
    """  Compute the half-life time
    inputs:
        nbb       : float, number of signal events in RoI
        exposure  : float, (kg y)
        eff       : total signal efficiency, (in fraction)
        abundance : float, isotope fraction (0.9)
        W         : float, Atomic weight, (135.9 g/mol for 136Xe)
    returns:
        tau       : float, half-life (y)
    """
    NA   = constants.Avogadro
    tau  = 1e3 * eff * abundance * (exposure / nbb) * (NA / W) * np.log(2.)
    return tau


def efficiencies(df, names = varnames, ranges = varranges):
    """ returns the efficiencies and its uncertatines for a serie of selections
    inputs:
        df    : DF
        names : tuple(str), names of the variables to select a given range
        ranges: tuple( (float, float), ), ranges of the variables
    returns:
        eff   : tuple(float), efficiency of each cut in the serie
        ueff  : tuple(float), uncertainty of the efficiency in each cut of the serie
    """

    sels = ut.selections(df, names, ranges)
    effs = [ut.efficiency(sel) for sel in sels]
    eff  = [x[0] for x in effs]
    ueff = [x[1] for x in effs]
    return eff, ueff

def selection(df, varnames, varranges, full_output = False):
    """
    
    Aply a list of selection in sequence.
    Returns the total selection or the sequence if full_output is True

    Parameters
    ----------
    df          : DataFrame, data
    varnames    : (tuple of string), list of the variables
    varranges   : (tuple of the ranges), accepted range for each variable
    full_output : bool, return the sequence of selections, default = False

    Returns
    -------
    sels : array(bool), total selection, if full_output is True a 
          list(array(bool)) with the consecutive selections

    """
    sel, sels =  None, []
    for i, varname in enumerate(varnames):
        isel = ut.in_range(df[varname].values, varranges[i])
        sel  = isel if sel is None else sel & isel
        sels.append(sel)
    out = sels if full_output else sel
    return out

#def selection_analysis(xdf,  
#                       xroi   = eroi, 
#                       eblob2 = eblob2):
#    sel = (xdf.num_tracks == 1) & (xdf.E >= xroi[0]) & \
#        (xdf.E < xroi[1]) & (xdf.blob2_E > eblob2)
#    return sel



def selection_blind(df, eroi = eroi, eblob2 = eblob2):
    """
    
    returns the blind selection

    Parameters
    ----------
    df     : DataFrame, data
    eroi   : tuple(float, float), Energy Track0 blind range
    eblob2 : tuple(float, float), Energy range of the 2nd blob

    Returns
    -------
    sel    : np.array(bool), selection of blind events

    """
    sel0 = (df.track0_E >= eroi[0]) &  (df.track0_E < eroi[1])
    sel1 = (df.blob2_E  > eblob2)
    sel  = np.logical_or(sel0, sel1)
    return ~sel

# def blind_mc_samples(mcs, blindvar = blindvar, blindrange = blindrange):
#     """ return MC sample with a blind region in the variable *blindvar* and range *blindrange*
#     inputs:
#         mcs        : tuple(DF), DFs with the MC samples
#         blindvar   : str, name of the variable used to blind
#         blindrange : tuple(float, float), range of the variable to blind
#     returns:
#         mcs        : tuple(DF), blind DFs
#     """
#     bmcs = []
#     for mc in mcs:
#         sel = ~ut.selection(mc, blindvar, blindrange)
#         bmcs.append(mc[sel])
#     return bmcs


def generate_mc_experiment(mcs, nevts, unevts = None):
    """ generate a MC experiment with the mcs samples, mcs, and the events in each sample, nevents
    inputs  :
        mcs     : tuple(DF), DFs with the mc samples
        nevents : tuple(int), number of events to select in each sample
        unevts  : tuple(float) or None, constrain the input events with a gaussian of sigma unevts
    returns :
        mc      : DF, DF with the mix of number event of events, nevents, of the initial samples, mcs
    """
    unevts = len(nevts) * [None,] if unevts == None else unevts
    #print(unevts)
    #nns  = [stats.poisson.rvs(ni, size = 1)[0] for ni in nevents]
    def _ni(ni, un):
        nk = ni if un == None else stats.norm.rvs(ni, un, size = 1)[0]
        return stats.poisson.rvs(nk, size = 1)[0]
    nns  = [_ni(ni, un) for ni, un in zip(nevts, unevts)]
    #print(nns)
    xmcs = [mc.sample(n = ni) for mc, ni in zip(mcs, nns)] # sample ni events of mc-sample mci
    mc   = pd.concat(xmcs) # concatenate all background to generate a mc sample that mimics the data-blind sample
    return mc


def fit_ell(data,
            mcs,
            ns,
            uns = None,
            varname = 'E',
            varrange = erange,
            bins = 150):
    """
    
    Fit to the number of events in samples, the varaible *varname* 
    in *varrange* using the data DF and as templates the mcs samples
    Use as initial parameters *ns* with optional uncertainties *uns*, 
    pdfs are binned in *bins* number of bins

    Parameters
    ----------
    data     : dataframe, data, include variable *varname*
    mcs      : tuple of DF, mc data frames to define the templates pdfs for each sample
    ns       : tuple of floats, the number of initual number of events to fit in each sample
    uns      : tuple of floats, optionnal, uncertainties of the number of events used as constrains, 
                    if False not contrain
    varname  : string, name of the variable to fit, default 'E'.
    varrange : (float, float), range of the variable to fit, default *erange*
    bins     : (int), number of bins of the pdfs, default is 100

    Returns
    -------
    result : object, results of the fit
    x      : array,  best paramerters - fit result
    ell    : object, Extended LikeLihood object, to plot likelihood for example,
             see *efit* module
    pdfs   : tuple(pdfs), templated pdfs for each sample

    """

    # generate pdfs from the MC
    #print('pdf variable : ', varname, varrange, bins)
    pdfs = [stats.rv_histogram(np.histogram(mc[varname], bins, range = varrange)) 
            for mc in mcs]

    # generate ELL object with the PDFs
    ell  = efit.ExtComPDF(pdfs, *ns) if uns ==  None else \
        efit.ConstrainedExtComPDF(pdfs, ns, uns)

    # get the data variable
    sel = ut.selection(data, varname, varrange)
    x   = data[varname][sel]

    # fit
    result = ell.best_estimate(x, *ns)
    #nsbest = result.x

    return result, x, ell


def plot_fit_ell(x,
                 par, 
                 ell, 
                 bins = 100,
                 parnames = ssamples,
                 plot_residuals = True):
    """ plot the data x, and superimposed the pdf with parameters (par)
    inputs:
        x    : np.array(float), data to plot
        par  : tuple(float), parameters of the pdf
        pdf  : function, pdf(x, *par), the pdf values of the distribution along x
        pdfs : tuple(pdfs), the mc pdfs for each sample
        parnames: tuple(str), list of the parameters (and samples) for the legend
    """

    subplot = pltext.canvas(1, 1, 8, 10)
    subplot(1)

    counts, edges = np.histogram(x, bins);
    centers = 0.5 * (edges[1:] + edges[:-1])
    ecounts = np.sqrt(counts)
    sel     = ecounts > 0
    nn      = np.sum(par)
    factor  = nn * (centers[1] - centers[0])
    plt.errorbar(centers[sel], counts[sel], yerr = ecounts[sel], 
                 marker = 'o', ls = '', label = 'data')

    label  = 'ELL fit \n'
    plt.plot(centers, factor * ell.pdf(centers, *par), label = label)

    i = 0
    for ni, ipdf in zip(par, ell.pdfs):
        factor = ni * (centers[1] - centers[0])
        label  = ' {:s} : {:6.2f} \n'.format(ssamples[i], ni)
        plt.plot(centers, factor * ipdf.pdf(centers), label = label)
        i += 1
    plt.legend(); plt.grid();

    if (not plot_residuals): return
    
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size = '20%', pad = 0)
    ax.figure.add_axes(ax2)
    fun = lambda x, *p : factor * ell.pdf(x, *p)
    pltext.hresiduals(x, bins, fun, par)

    return


def ana_samples(data, 
                mcs, 
                varnames   = varnames[:-1],
                varranges = varranges[:-1],
                mc_level   = -1,
                verbose    = True):
    """
    
    Return the sample to analysis after a selection in the variables *varnames* 
    in ranges *varranges*

    Parameters
    ----------
    data      : DataFrame, data
    mcs       : tuple(DataFrame), mc data frames for the different samples
    varnames  : tuple(float), list of the variables names of the selection
    varranges : tuple( (float, float)), list of the variable ranges of the selection
    mc_level  : int, reduce the selection for mc samples to get more stats for the pdfs
                a given level, default = -1
    verbose   : bool, print into

    Returns
    -------
    anadata   : DataFrame, the selected data events
    anamcs    : tuple(DataFrame), the selected mc events (to get the pdfs)
    effs      : tuple(float), selection efficiencies in the different samples

    """
    
    effs    = [efficiencies(mc, varnames, varranges)[0][-1]  for mc in mcs]
    anamcs  = [mc[selection(mc, varnames[:mc_level], varranges[:mc_level])] for mc in mcs]
    anadata = data[selection(data, varnames, varranges)]
    
    if (verbose):
        print('selection variables  :', varnames)
        print('selection ranges     :', varranges)
        print('selection mc samples :', varnames[:mc_level])
        print('data size            :', len(anadata))
        print('mc sizes             :', [len(mc) for mc in anamcs])
        print('efficiencies         : {:6.2f}, {:1.2e}, {:1.2e}'.format(* effs))
    
    return anadata, anamcs, effs


def plot_contributions(data,
                       mcs,
                       ns,
                       varname  = 'E',
                       varrange = erange,
                       nbins    = 80,
                       ssamples = ssamples):
    
    # plot data
    counts, bins = np.histogram(data[varname], nbins, range = varrange)
    cbins = 0.5 * (bins[1:] + bins[:-1])
    esel  = counts > 0
    ecounts = np.sqrt(counts)
    plt.errorbar(cbins[esel], counts[esel], yerr = ecounts[esel], marker = 'o', ls = '', label = 'data');
    plt.ylabel('counts')
        
    #ax = plt.gca().twinx()
    i = 0
    utots = np.zeros(len(counts))
    for n, mc in zip(ns, mcs):
        ucounts, _   = np.histogram(mc[varname], bins)
        ucounts      = n * ucounts/np.sum(ucounts)
        utots        += ucounts
        plt.plot(cbins, ucounts, label = ssamples[i]);
        i += 1
    plt.plot(cbins, utots, label = 'total');
    plt.grid(); plt.title(varname), plt.legend();
    return


def ana_experiment(data,
                   mcs,
                   nevts,
                   unevts    = None,
                   varnames  = varnames[:-1],
                   varranges = varranges[:-1],
                   mc_level  = -1,
                   varname   = 'E',
                   varrange  = erange,
                   bins      = 150,
                   verbose   = True): 
    """

    perform the analysis

    Parameters
    ----------
    data   : DataFrame, data
    mcs    : tuple(DataFrame), mc data (bb, Bi, Tl)
    nevts  : tuple(float), number of expected total events of each sample
    unevts : tuple(float), optional, uncertainties of the expected total events in each sample
        if provided, the fit is constrained
    varnames  : tuple(strig), list of the variables of the selection
    varranges : tuple( (float, float)), list with the selection variable ranges
    mc_level  : int, optional, the mc samples for the pdfs use a 'downscaled' selection 
    varname   : float, variable of the fit. The default is 'E'.
    varrange  : (float, float), range of the fit. The default is erange.
    bins      : int, optional, number of bins. The default is 150.
    verbose   : bool, optional, print out and plots. The default is True.

    Returns
    -------
    result    : object, result of the fit with the LL, and the estimated parameters
    enes      : np.array(float), data values entered into the fit
    ell       : object, extended LL fit object, access to the loglike
    pdfs      : tuple(object), pdf objects associated to each sample

    """
    
    anadata, anamcs, effs = ana_samples(data, mcs, varnames, varranges, 
                                        mc_level = mc_level, verbose = verbose)
    nns  = [eff * ni  for eff, ni  in zip(effs, nevts)]
    unns = None if unevts == None else [eff * uni for eff, uni in zip(effs, unevts)]
    result, enes, ell  = fit_ell(anadata, anamcs, nns, unns,
                                 varname = varname, varrange = varrange, 
                                 bins = bins)
    
    if (verbose):
        print('Initial       Events : {:6.2f}, {:6.2f}, {:6.2f}'.format(* nns))
        if (unns != None):
            print('Uncertainties Events : {:6.2f}, {:6.2f}, {:6.2f}'.format(*unns))
        print('Fit success          : ', result.success)
        print('Estimated     Events : {:6.2f}, {:6.2f}, {:6.2f}'.format(*result.x))
        plot_fit_ell(enes, result.x, ell)
    
    return result, enes, ell
    

#----------------
#
# def test_pdf(dat, mcs, varname, bins, varrange, ssamples = ssamples):
#     ene  = dat[varname].values
#     ene  = ene[ut.in_range(ene, erange)]
#     print('events', len(ene))
#
#     subplot = pltext.canvas(2)
#
#     subplot(1)
#     pltext.hist(ene, bins, label = 'blind data');
#     plt.xlabel(varname);
#
#     subplot(2)
#     for i, mc in enumerate(mcs):
#         pltext.hist(mc[varname], bins, range = erange, density = True, label = ssamples[i])
#     plt.xlabel(varname);
