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

def selection_analysis(xdf,  
                       xroi   = eroi, 
                       eblob2 = eblob2):
    sel = (xdf.num_tracks == 1) & (xdf.E >= xroi[0]) & \
        (xdf.E < xroi[1]) & (xdf.blob2_E > eblob2)
    return sel



def selection_blind(df, eroi = eroi, eblob2 = eblob2):
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


def generate_mc_experiment(mcs, nevents):
    """ generate a MC experiment with the mcs samples, mcs, and the events in each sample, nevents
    inputs  :
        mcs     : tuple(DF), DFs with the mc samples
        nevents : tuple(int), number of events to select in each sample
    returns :
        mc      : DF, DF with the mix of number event of events, nevents, of the initial samples, mcs
    """
    xmcs = [mc.sample(n = int(ni)) for mc, ni in zip(mcs, nevents)] # sample ni events of mc-sample mci
    mc    = pd.concat(xmcs) # concatenate all background to generate a mc sample that mimics the data-blind sample
    return mc


def fit_ell(data, mcs, ns, varname = 'E', bins = 100, varrange = erange):
    """ Fit the variable 'varname' ('E') of the DF of the data (data) to a combined PDF
    of the MC samples (mcs), using a histogram with number of bins (bins) in the range of
    the variable (varrange)
    inputs:
        data    : DF, DF of the data
        mcs     : tuple(DF), list of mcs samples DFs
        varname : str, name of the variable to fit, to generate the combined PDF
        varrange: (float, float), range of the histogram of the PDFs
    returns:
        results : A FitResult object with the fit results (see optimize.minimize in Scipy)
        x       : the data used in the fit,
                  that is the values of the data varialbe (varname) in range (varrange)
        ell     : ExtComPDF object, An object with the functionality of a Extended Combined PDF
                  (see efit module)
    """

    # generate pdfs from the MC
    pdfs = [stats.rv_histogram(np.histogram(mc[varname], bins, range = varrange))
            for mc in mcs]

    # generate ELL object with the PDFs
    ell  = efit.ExtComPDF(pdfs, *ns)

    # get the data variable
    sel = ut.selection(data, varname, varrange)
    x   = data[varname][sel]

    # fit
    result = ell.best_estimate(x, *ns)
    #nsbest = result.x

    return result, x, ell


def plot_fit_ell(x, par, pdf, bins = 100, parnames = ssamples):
    """ plot the data x, and superimposed the pdf with parameters (par)
    inputs:
        x   : np.array(float), data to plot
        par : tuple(float), parameters of the pdf
        pdf : function, pdf(x, *par), returns the pdf values of the distribution along x
        parnames: tuple(str), list of the parameters names to write them in the plot legend
    """

    subplot = pltext.canvas(1, 1, 8, 10)
    subplot(1)

    counts, edges = np.histogram(x, bins);
    centers = 0.5 * (edges[1:] + edges[:-1])
    ecounts = np.sqrt(counts)
    sel     = ecounts > 0
    nn     = np.sum(par)
    factor = nn * (centers[1] - centers[0])
    plt.errorbar(centers[sel], counts[sel], yerr = ecounts[sel], marker = 'o', ls = '', label = 'data')

    label  = 'ELL fit \n'
    for si, ni in zip(parnames, par):
        label += ' {:s} : {:6.2f} \n'.format(si, ni)
    plt.plot(centers, factor * pdf(centers, *par), label = label)
    plt.legend(); plt.grid();

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size = '20%', pad = 0)
    ax.figure.add_axes(ax2)
    fun = lambda x, *p : factor * pdf(x, *p)
    pltext.hresiduals(x, bins, fun, par)

    return



def ana_experiment(data, mcs, nevts, level_data = 2):

    #level_mc = 1 if selection is None else level_mc
    #print(level_mc)
    #vnames, vranges = (varnames[:-1], varranges[:-1]) if selection is None else selection

    vnames   = ['E', 'num_tracks', 'blob2_E']
    vranges  = [(2.400, 2.650), (1., 1.1), (0.4, np.inf)]
    level_mc = min(1, level_data) # MC level selction E in range, 1 track

    sels = [ut.selections(mc, vnames, vranges) for mc  in mcs]
    effs = [ut.efficiency(sel[level_data])     for sel in sels]
    tmcs = [mc[sel[level_mc]]                  for mc, sel in zip(mcs, sels)]
    tdat = data[ut.selections(data, vnames, vranges)[level_data]]

    # expected number of events after selection
    ns_exp       = [ni * eff[0] for ni, eff in zip(nevts, effs)]
    res, x, ell  = fit_ell(tdat, tmcs, ns_exp)

    return (res, x, ell, effs)






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
