import warnings

import numpy as N
from scipy import stats
import h5py

try:
    import pocomc
except ImportError:
    pocomc = None

from .fit import Likelihood
from .par import Par, Pars, PriorFlat, PriorGaussian, PriorFlatSoft, PriorBoundedGaussian, \
    PriorBoundedGaussianSoft


class PocoMCSampler:
    """A sampler wrapper of PocoMC (https://pocomc.readthedocs.io/)

    :param Fit fit: mbproj2d Fit object
    :param dict sampler_config: configuration dictionary passed to pocomc.Sampler()
    """

    def __init__(self, fit, sampler_config=None):
        if pocomc is None:
            raise RuntimeError("Package pocomc missed")

        self.fit = fit
        if sampler_config is None:
            sampler_config = {}

        self._loglike_creation()  # a dirty way to create a global picklable function pocomc_loglike 

        prior = self._prior_creation(self.fit.pars.copy())

        self.sampler = pocomc.Sampler(prior=prior,
                                      likelihood=pocomc_loglike,
                                      **sampler_config
                                      )

    def _prior_creation(self, pars):
        """
        :param Pars pars: mbproj2d Pars object
        """
        pars: Pars

        prior_rv_list = []
        for par_name in pars.freeKeys():
            par: Par = pars[par_name]
            prior_rv_list += [self.create_rv_from_mbprior(par.prior)]

        pocomc_prior = pocomc.Prior(prior_rv_list)

        return pocomc_prior

    def _loglike_creation(self):
    
        pars = self.fit.pars.copy()

        global pocomc_loglike

        def pocomc_loglike(vals):
            pars.setFree(vals)
            like = Likelihood(self.fit.images, self.fit.model, pars)
            return N.sum(like.images)


    def run(self, output_file, run_config=None):
        """
        Run sampler and save posterior samples

        :param output_file:
        :param run_config: a configuration dictionary including parameters for pocomc.Sampler.run
        """

        if run_config is None:
            run_config = {}


        # run sampler
        self.sampler.run(**run_config)

        # generate posterior samples + log_like
        resampled_points, log_like, log_prior = self.sampler.posterior(resample=True)

        # To keep using the current mcmc chain structure,
        # the resampled_points will be reshpaed to a [Nstep, 1, Npara] array
        #
        with h5py.File(output_file, "w") as f:
            f["chain"] = N.reshape(resampled_points, [resampled_points.shape[0], 1, resampled_points.shape[1]])
            f["likelihood"] = N.reshape(log_like, [log_like.size, 1])
            f["thawed_params"] = N.array([x.encode('utf-8') for x in self.fit.pars.freeKeys()])

    @staticmethod
    def create_rv_from_mbprior(prior):
        """converting an Mbproj2D prior to a scipy rv

        "param PriorBase prior": mbproj2d prior
        """

        if isinstance(prior, (PriorFlat, PriorFlatSoft)):
            rv = stats.uniform(prior.minval, prior.maxval - prior.minval)
        elif isinstance(prior, PriorGaussian):
            rv = stats.norm(prior.mu, prior.sigma)
        elif isinstance(prior, (PriorBoundedGaussian, PriorBoundedGaussianSoft)):
            warnings.warn(f"{prior.__class__} not supported yet, use gaussian prior instead.")
            rv = stats.norm(prior.mu, prior.sigma)
        else:
            raise TypeError("Input prior type invalid.")

        return rv
