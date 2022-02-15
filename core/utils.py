import numpy             as np


#--- general utilies

def remove_nan(vals : np.array) -> np.array:
    """ returns the np.array without nan
    """
    return vals[~np.isnan(vals)]


def in_range(vals : np.array, range : tuple = None, upper_limit_in = False) -> np.array(bool):
    """ returns a np.array(bool) with the elements of val that are in range
    inputs:
        vals : np.array
        range: tuple(x0, x1)
    returns
        np.array(bool) where True/False indicates if the elements of vals is in rage
    """
    if (range is None): return vals >= np.min(vals)
    sel1 = (vals >= range[0])
    sel2 = (vals <= range[1]) if upper_limit_in else (vals < range[1])
    return sel1 & sel2

def centers(xs : np.array) -> np.array:
    """ returns the center between the participn
    inputs:
        xs: np.array
    returns:
        np.array with the centers of xs (dimension len(xs)-1)
    """
    return 0.5* ( xs[1: ] + xs[: -1])


# def arscale(x, scale = 1.):
#     """ return an array between [0., 1.]
#     inputs:
#         x    : np.array,
#         scale: float (1.)
#     returns:
#         np.arry with scaled balues, [0, scale]
#     """
#     xmin, xmax = np.min(x), np.max(x)
#     rx = scale * (x - xmin)/(xmax - xmin)
#     return rx
#
#
# def arstep(x, step, delta = False):
#     """ returns an array with bins of step size from x-min to x-max (inclusive)
#     inputs:
#         x    : np.array
#         step : float, step-size
#     returns:
#         np.array with the bins with step size
#     """
#     delta = step/2 if delta else 0.
#     return np.arange(np.min(x) - delta, np.max(x) + step + delta, step)


def stats(vals : np.array, range : tuple = None):
    vals = np.array(vals)
    vals = remove_nan(vals)
    sel  = in_range(vals, range)
    vv = vals[sel]
    mean, std, evts, oevts = np.mean(vv), np.std(vv), len(vv), len(vals) - len(vv)
    return evts, mean, std, oevts


def str_stats(vals, range = None, formate = '6.2f'):
    evts, mean, std, ovts = stats(vals, range)
    s  = 'entries '+str(evts)+'\n'
    s += (('mean {0:'+formate+'}').format(mean))+'\n'
    s += (('std  {0:'+formate+'}').format(std))
    return s


#-----

def efficiency(sel, n = None):
    """ compute the efficiency and uncertantie of a selection
    inputs:
        sel: np.array(bool), bool array with True/False
        n  : int, denominator, if n is None, use len(sel)
    """
    n    = n if n is not None else len(sel)
    eff  = np.sum(sel)/n
    ueff = np.sqrt(eff * (1- eff) / n)
    return eff, ueff


def selection(df, varname, varrange):
    """ apply the selection df.varname in a range, varange
    inputs:
        df       : dataFrame
        varname  : str, name of the variable int he DF
        varrange : tuple(float, float), range of the selection, all (-np.inf, np.inf)
    returns:
        sel      : np.array(bool) same same of DF with True/False
                   if the item fulfull variable value (varname) inside the range (varrange)
    """
    sel = in_range(df[varname].values, varrange)
    return sel


def selections(df, varnames, varranges):
    """ appy a list of selections in order in a data-frame
    inputs:
        df        : DataFrame
        varnames  : tuple(str), list of the variables of the selections
        varranges : tuple(float, float), range of the variables in the selection
    returns:
        sels: tuple(np.array(bool)), tuple with the selections applied in series
    """
    sel, sels =  None, []
    for i, varname in enumerate(varnames):
        isel = selection(df, varname, varranges[i])
        sel  = isel if sel is None else sel & isel
        sels.append(sel)
    return sels


def sample_selection(df, varname, varrange):
    sel = selection(df, varname, varrange)
    sdf = df[sel]
    return sdf, sel

def sample_selections(df, varnames, varranges):
    sel = selections(df, varnames, varranges)[-1]
    sdf = df[sel]
    return sdf, sel