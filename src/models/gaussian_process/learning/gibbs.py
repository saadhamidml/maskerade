from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel


def gibbs_sample():

    def gibbs_model(params: Mapping) -> Mapping:
        params
        for j in range(n_random_fourier_features):
            # Draw Mixture component, z_j
            z_j = params[f'z_{j}']
            params[f'z_{j}'] = new_z_j

        for k in range(n_random_fourier_features):
            # Draw mu_k
            mu_k = params[f'mu_{k}']
            params[f'mu_{k}'] = new_mu_k

            # Draw sigma_k
            sigma_k = params[f'sigma_{k}']
            params[f'sigma_{k}'] = new_sigma_k

        for j in range(n_random_fourier_features):
            # Draw random frequency, w_j
            w_j = params[f'w_{j}']
            params[f'w_{j}'] = new_w_j

        return new_params

    initial_params = intial_samples()


    gibbs_kernel = Gibbs(gibbs_model)
    mcmc = MCMC(
        gibbs_kernel, num_samples=num_samples, initial_params=initial_params
    ).run(data)

    return mcmc


class Gibbs(MCMCKernel):
    def __init__(self, model=None, *args, **kwargs):
        super(Gibbs, self).__init__(*args, **kwargs)
        self.model = model

    def sample(self, params):
        """params is dictionary."""
        return self.model(params)
