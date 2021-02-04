#
# Classes and methods to fit samples from a combination of pdfs
# using the extended likelihood
#
# JAH, 25/1/2021

import numpy as np
import scipy.stats    as stats
import scipy.optimize as optimize


def loglike_uncertainties(fun, par_est, scale = 0.01, bounds = None):

    ## TODO: the likeihood scan requires to fit the noisance parameters!
    npars  = len(par_est)
    ts_est = fun(par_est)
    par_bounds = npars*((-np.inf, np.inf),) if bounds is None else bounds
    assert (len(par_bounds) == npars), 'required equal number of parameters and bounds ranges'

    def _u(i):

        mask    = np.full(npars, False)
        mask[i] = True
        pi      = par_est[mask]
        sigma   = np.abs(pi * scale)

        def _fun(pi):
            cpar    = np.copy(par_est)
            cpar[i] = pi
            val     = (fun(cpar) - ts_est - 1)
            return val * val

        bounds = ((pi, par_bounds[i][1]),)
        res    = optimize.minimize(_fun, pi + sigma, bounds = bounds)
        x_up   = res.x[0] if res.success else np.NAN
        bounds = ((par_bounds[i][0], pi),)
        res    = optimize.minimize(_fun, pi - sigma, bounds = bounds)
        x_down = res.x[0] if res.success else np.NAN
        return (x_down, x_up)

    res = [_u(i) for i in range(npars)]
    return res


#def parameters_from_fitresult(res):
#    assert res.success == True, 'Only valid Fit Results'
#    par    = res.x
#    upar   = res.uncertainties
#    sigmas = [np.mean(abs(ui-p)) for p, ui in zip(par, upar)]
#    return par, sigmas, upar


class ComPDF:
    """ A Stats class for a PDF obtained with the composition of different PDFs.
    each one enters into the combined PDF with a given weight.
    """

    def __init__(self, pdfs, *ns):
        """ constructors:
        inputs:
            pdfs: list(pdf), pdf must have the pdf(x), logpdf(x), rvs(size), methods implemented
                  these methods compute the pdf and generate randon variables.
            ns  : list(float), the weights of each pdf, or number of events of each pdf.
                  Internally ns will be converted to weights ns/sum(ns) to ensure normalization.
                  default: None, all pdfs enter with the same weight 1.
        """
        self.pdfs = pdfs
        self.ns   = np.array(ns) if len(ns) > 0 else np.ones(len(pdfs))
        assert len(self.ns) == len(self.pdfs), 'number of pdfs and number of samples must be the same'


    def weights(self, *ns):
        """ return the normalization, sum(ns), and the weights of each pdf
        returns:
            ntot, float, sum of the number of events in each sample
            fis, tuple(float), weight of each sample, n_i/ntot
        """
        ns = ns if len(ns) > 0 else self.ns
        assert len(self.ns) == len(self.pdfs), 'number of pdfs and number of samples must be the same'
        ntot = np.sum(ns)
        fis  = ns/ntot
        return ntot, fis


    def pdf(self, x, *ns):
        """ pdf value of x given ns list of events
        (if ns is empty, the weights in the self object will be used)
        """
        ntot, fis = self.weights(*ns)
        for fi in fis:
            if np.any(fi < 0.): return np.inf
        p = [fi * ipdf.pdf(x) for fi, ipdf in zip(fis, self.pdfs)]
        p = np.sum(p, axis = 0)
        return p


    def loglike(self, x, *ns):
        """ return the log-likelihood of a x - observation
        inputs:
            x , array(float), data
            ns, tuple(float), number of expected events in each sample
                if empty, the values stored in the object, self, will be used
        """
        return np.sum(np.log(self.pdf(x, *ns)))


    def best_estimate(self, x, *ns):
        """ best estimate of the number of events, ns, from the data x,
            obtained by minimising -2 log likelihood.
        inputs:
            x  : np.array(float), data
            ns : tuple(float), list with the initial guess of the number of events in each sammple
        returns:
            res : A ResultFit object (see minimize function in scipy module otimize)
        """

        m, fi = self.weights(*ns)
        ns    = m * fi

        fun = lambda par: -2. * self.loglike(x, *par)
        bounds = len(ns) * ((0., np.inf),)

        res = optimize.minimize(fun, ns, bounds = bounds)
        return res


    def rvs(self, *ns, size = 1):
        """ generate size random variables x using the ns number of events
        inputs:
            ns, tuple(float), number of events in each sample
            (if ns is empty, the weights in the self object will be used)
            size, int, (default 1), size of the random data
        returns:
            x. array(float), of size, with the random generated data
        """
        ntot, fis = self.weights(*ns)
        cfis = [np.sum(fis[: i + 1]) for i in range(len(fis))]
        u  = np.random.uniform(size = size)
        iu = np.digitize(u, cfis)
        x  = np.array([self.pdfs[i].rvs() for i in iu])
        return x


class ExtComPDF(ComPDF):
    """ class to do a Extended Like-Lihood fit to a PDF made from a compositions of PDFs (pdfs),
    each one with an expected number of events (ns)
    """

    def __init__(self, pdfs, *ns):
        """ constructor with a list of pdfs (must have the method, pdf, logpdf, rvs)
        and the expected number of events for each pdf
        inputs:
            pdfs: list(pdf), pdf is an PDF object with methods pdf, logpdf, and rvs
            ns  : list(float), number of expected events in each pdf
        """
        ComPDF.__init__(self, pdfs, *ns)
        #elf.compdf = PDFComposite(pdfs, *ns)
        #return


    def loglike(self, x, *ns):
        """ return extended -2 log likelihood, for x-data, ns is the list of expected number of events of each sample,
        inputs:
            x, np.array(float), data
            ns, tuple(float), number of events in each sample
                (is ns is empty, the values in the self object will be used)
        """
        lp1  = ComPDF.loglike(self, x, *ns)
        n    = len(x)
        m, _ = self.weights(*ns)
        lp2  = stats.poisson.logpmf(n, m)
        return (lp1 + lp2)


    def rvs(self, *ns, size = 1):
        """ generate size experiments, each one will have n-poisson distrubted events
        generated with the combined PDF with weights according with ns.
        inputs:
            ns  : tuple(float), number of expected evens in each sample
            size: int, number of experiments
                 (not that each experiment is an array of n-poisson distributed entries)
        returns:
            x   : np.array, list of array with the events generated in eqch experiment.
                 (if size = 1) only one array with the data of the experiment
        """

        m, _  = self.weights(*ns)
        def _rv():
            n     = stats.poisson.rvs(size = 1, mu = m)
            x     = ComPDF.rvs(self, *ns, size = n)
            return x

        xs = [_rv() for i in range(size)]
        xs = xs[0] if size == 1 else xs
        return xs
