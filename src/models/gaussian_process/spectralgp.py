"""Modified version of FKL to allow for tracking progress on the test set."""
import torch
import gpytorch
import copy
import time
from spectralgp.samplers import AlternatingSampler, EllipticalSliceSampler, SGD

from .inference import evaluate_functional_kernel_learning


class TimedAlternatingSampler(AlternatingSampler):
    def __init__(self, *args, learning_time_limit=None, **kwargs):
        super(TimedAlternatingSampler, self).__init__(*args, **kwargs)
        self.learning_time_limit = learning_time_limit

    def run(
            self,
            test_x=None,
            test_y=None,
            live_inference=False,
            sacred_run=None
    ):
        total_time = 0.0

        outer_samples = [[] for x in range(self.num_dims)]
        inner_samples = [[] for x in range(self.num_dims)]

        if live_inference:
            log_probs = torch.tensor([]).to(self.model[0].train_inputs[0])
            means = torch.zeros(test_y.size(0)).unsqueeze(0).to(
                self.model[0].train_inputs[0]
            )
        else:
            log_probs = None
            means = None
        for step in range(self.totalSamples):
            ts = time.time()
            for in_dim in range(self.num_dims):

                if self.num_dims == 1:
                    idx = None
                else:
                    idx = in_dim
                    print('Step: ', step, 'Dimension: ', in_dim)

                # run outer sampler factory
                curr_outer_samples, _ = self.outer_sampler_factory(self.numOuterSamples,
                                                                   self.model, self.likelihood,
                                                                   idx, sacred_run=sacred_run).run()

                # loop through every task
                curr_task_list = []
                for task in range(self.num_tasks):
                    print('Task:', task, "; Iteration", step)
                    # run inner sampler factory
                    with torch.no_grad():
                        curr_task_samples, _, log_probs, means, inference_time = self.inner_sampler_factory[task](
                            self.numInnerSamples,
                            self.model[task],
                            self.likelihood[task],
                            idx,
                            sacred_run=sacred_run
                        ).run(
                            model=self.model[0],
                            log_probs=log_probs,
                            means=means,
                            test_x=test_x,
                            test_y=test_y,
                            live_inference=False,
                            sacred_run=sacred_run,
                        )

                        curr_task_list.append(copy.deepcopy(curr_task_samples.unsqueeze(0)))

                curr_inner_samples = torch.cat(curr_task_list, dim=0)

                outer_samples[in_dim].append(copy.deepcopy(curr_outer_samples))
                inner_samples[in_dim].append(copy.deepcopy(curr_inner_samples))

            ts_d = torch.abs(torch.tensor(time.time() - ts)).item()
            total_time += ts_d
            print("Seconds for Iteration {} : {}".format(step, ts_d))

            if live_inference and sacred_run is not None:
                start_time_i = time.time()
                try:
                    from copy import deepcopy
                    model_copy = deepcopy(self.model[0])
                    model_copy.eval()
                    self.gsampled = [
                        torch.cat(inner_samples[id], dim=-1)
                        for id in range(self.num_dims)
                    ]
                    ll, rmse, _, _ = evaluate_functional_kernel_learning(
                        test_x, test_y, model_copy, self, 5
                    )
                    del model_copy
                    sacred_run.log_scalar(
                        metric_name='test_data_log_marginal_likelihood',
                        value=ll
                    )
                    sacred_run.log_scalar(
                        metric_name='test_rmse',
                        value=rmse
                    )
                    sacred_run.log_scalar(
                        metric_name='elapsed_time',
                        value=total_time
                    )
                except:
                    pass
                inference_time = time.time() - start_time_i
                total_time -= inference_time

            if self.learning_time_limit is not None:
                if total_time > self.learning_time_limit:
                    if sacred_run is not None:
                        sacred_run.log_scalar(
                            metric_name='learning_time',
                            value=total_time
                        )
                    break

        self.hsampled = [torch.cat(outer_samples[id], dim=-1) for id in range(self.num_dims)]
        self.gsampled = [torch.cat(inner_samples[id], dim=-1) for id in range(self.num_dims)]
        # return self.hsampled, self.gsampled # don't return anything, makes notebooks icky


class InferringEllipticalSliceSampler(EllipticalSliceSampler):
    def __init__(self, *args, **kwargs):
        super(InferringEllipticalSliceSampler, self).__init__(*args, **kwargs)

    def run(
        self,
        model=None,
        log_probs=None,
        means=None,
        test_x=None,
        test_y=None,
        live_inference=False,
        sacred_run=None
    ):
        inference_time = 0.
        start_time = time.time()
        print("Starting ess sampling")
        self.f_sampled = torch.zeros(self.n, self.n_samples)
        self.ell = torch.zeros(self.n_samples, 1)

        f_cur = self.f_init
        for ii in range(self.n_samples):
            if ii == 0:
                ell_cur = self.lnpdf(f_cur, *self.pdf_params)
            else:
                f_cur = self.f_sampled[:, ii - 1]
                ell_cur = self.ell[ii - 1, 0]

            next_f_prior = self.f_priors[:, ii]

            self.f_sampled[:, ii], self.ell[ii] = self.elliptical_slice(f_cur, next_f_prior,
                                                                        cur_lnpdf=ell_cur,
                                                                        pdf_params=self.pdf_params)
            if live_inference:
                # Update log prob sum
                start_time_ii = time.time()
                n_dimensions = test_x.size(1) if test_x.ndimension() > 1 else 1

                torch.cuda.empty_cache()
                try:
                    from copy import deepcopy
                    model_copy = deepcopy(model)
                    model_copy.eval()
                    if n_dimensions == 1:
                        last_sample = self.f_sampled[:, ii]
                        current_latent_params = model_copy.covar_module.get_latent_params()
                        model_copy.covar_module.set_latent_params(last_sample)
                        model_copy.prediction_strategy = None
                        model_output = model_copy.likelihood(model_copy(test_x))
                        with gpytorch.settings.max_cg_iterations(5000):
                            log_prob = model_output.log_prob(test_y).detach()
                        log_probs = torch.cat((log_probs, log_prob.unsqueeze(0)), dim=0)
                        predictive_log_likelihood = (
                            log_probs[-10:].logsumexp(dim=0)
                            - torch.tensor(log_probs[-10:].size(0)).to(test_y).log()
                        ).item()
                        means = torch.cat(
                            (means, model_output.mean.unsqueeze(0)), dim=0
                        )
                        rmse = (
                            means[-10:].mean(dim=0) - test_y
                        ).pow(2).mean().sqrt().item()
                        if sacred_run is not None:
                            sacred_run.log_scalar(
                                metric_name='test_data_log_marginal_likelihood',
                                value=predictive_log_likelihood
                            )
                            sacred_run.log_scalar(
                                metric_name='test_rmse',
                                value=rmse
                            )
                        model_copy.covar_module.set_latent_params(current_latent_params)
                    else:
                        # data_mod_means = torch.zeros_like(model_copy(test_x).mean)
                        # total_variance = torch.zeros_like(model_copy.likelihood(model_copy(test_x)).variance)
                        # with torch.no_grad():
                        #     # marg_samples_num = min(len(alt_sampler.fhsampled[0][0]), alt_sampler.fgsampled[0].shape[-1])
                        #     marg_samples_num = alt_sampler.fgsampled[0].shape[-1]
                        #     for x in range(0, marg_samples_num):
                        #         for dim in range(0, n_dimensions):
                        #             model_copy.covar_module.set_latent_params(
                        #                 alt_sampler.fgsampled[dim][0, :, x], idx=dim)
                        #         model_copy.set_train_data(model_copy.train_inputs[0], model_copy.train_targets)  # to clear out the cache
                        #         data_mod_means += model_copy(test_x).mean
                        #         y_preds = model_copy.likelihood(model_copy(test_x))
                        #         # y_var = f_var + data_noise
                        #         y_var = y_preds.variance
                        #         total_variance += (y_var + torch.pow(model_copy(test_x).mean, 2))
                        # meaned_data_mod_means = data_mod_means / float(marg_samples_num)
                        # total_variance = total_variance / float(marg_samples_num) - torch.pow(
                        #     meaned_data_mod_means, 2)
                        pass
                    del model_copy
                except RuntimeError:
                    from numpy import nan
                    if sacred_run is not None:
                        sacred_run.log_scalar(
                            metric_name='test_data_log_marginal_likelihood',
                            value=nan
                        )
                    model.train()
                torch.cuda.empty_cache()
                inference_time += time.time() - start_time_ii
            else:
                log_probs = None
                means = None
            print("FKL sample", ii , time.time() - start_time)
        return self.f_sampled, self.ell, log_probs, means, inference_time


class MeanEllipticalSlice(InferringEllipticalSliceSampler):
    def __init__(self, f_init, dist, lnpdf, nsamples, pdf_params=()):

        mean_vector = dist.mean

        demeaned_lnpdf = lambda g: lnpdf(g + mean_vector, *pdf_params)

        demeaned_init = f_init - mean_vector

        samples = dist.sample(sample_shape = torch.Size((nsamples,))).t()
        demeaned_samples = samples - mean_vector.unsqueeze(1)

        super(MeanEllipticalSlice, self).__init__(demeaned_init, demeaned_samples, demeaned_lnpdf, nsamples, pdf_params=())

        self.mean_vector = mean_vector

    def run(
            self,
            model=None,
            log_probs=None,
            means=None,
            test_x=None,
            test_y=None,
            live_inference=False,
            sacred_run=None
    ):
        self.f_sampled, self.ell, log_probs, means, inference_time = super().run(
            model=model,
            log_probs=log_probs,
            means=means,
            test_x=test_x,
            test_y=test_y,
            live_inference=live_inference,
            sacred_run=sacred_run
        )

        #add means back into f_sampled
        self.f_sampled = self.f_sampled + self.mean_vector.unsqueeze(1)

        return self.f_sampled, self.ell, log_probs, means, inference_time


# defining ESS factory
def ess_factory(nsamples, data_mod, data_lh, idx=None, sacred_run=None):
    # pull out latent model and spectrum from the data model
    omega = data_mod.covar_module.get_omega(idx)
    g_init = data_mod.covar_module.get_latent_params(idx)
    latent_lh = data_mod.covar_module.get_latent_lh(idx)
    latent_mod = data_mod.covar_module.get_latent_mod(idx)

    # update training data
    latent_lh.train()
    latent_mod.train()
    latent_mod.set_train_data(inputs = omega, targets = None, strict = False)

    # draw prior prior distribution
    prior_dist = latent_lh(latent_mod(omega))

    # define a function of the model and log density
    def ess_ell_builder(demeaned_logdens, data_mod, data_lh):
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False):

            data_mod.covar_module.set_latent_params(demeaned_logdens, idx)
            data_mod.prediction_strategy = None

            loss = data_lh(data_mod(*data_mod.train_inputs)).log_prob(data_mod.train_targets).sum()
            if sacred_run is not None:
                sacred_run.log_scalar(
                    metric_name='likelihood_evaluation',
                    value=float(loss),
                )
            return loss

    # creating model
    return MeanEllipticalSlice(g_init, prior_dist, ess_ell_builder, nsamples, pdf_params=(data_mod, data_lh))

# defining slice sampler factory
def ss_factory(nsamples, data_mod, data_lh, idx = None, sacred_run=None):
    if isinstance(data_mod, list):
        data_mod = data_mod[0]
        data_lh = data_lh[0]

    # defining log-likelihood function
    data_mod.train()
    data_lh.train()

    # pull out latent model and spectrum from the data model
    latent_lh = data_mod.covar_module.get_latent_lh(idx)
    latent_mod = data_mod.covar_module.get_latent_mod(idx)
    omega = data_mod.covar_module.get_omega(idx)
    demeaned_logdens = data_mod.covar_module.get_latent_params(idx)

    # update the training inputs
    latent_mod.set_train_data(inputs=omega, targets=demeaned_logdens.detach(), strict=False)

    data_mll = gpytorch.ExactMarginalLogLikelihood(data_lh, data_mod)

    def ss_ell_builder(latent_mod, latent_lh, data_mod, data_lh):

        latent_lh.train()
        latent_mod.train()
        # TODO: see what the runtime and results look like
        with gpytorch.settings.max_preconditioner_size(15), gpytorch.settings.cg_tolerance(1e-3), gpytorch.settings.max_cg_iterations(1000):
            loss = data_mll(data_mod(*data_mod.train_inputs), data_mod.train_targets)
            print('Loss is: ', loss)
            if sacred_run is not None:
                sacred_run.log_scalar(
                    metric_name='likelihood_evaluation',
                    value=float(loss),
                )
            #num_y = len(data_mod.train_targets)
            #print('P_y is: ', data_lh(data_mod(*data_mod.train_inputs)).log_prob(data_mod.train_targets)/num_y)
            #print('p_nu is: ', data_mod.covar_module.latent_prior.log_prob(data_mod.covar_module.latent_params)/num_y)
            return loss

    ell_func = lambda h: ss_ell_builder(latent_mod, latent_lh, data_mod, data_lh)

    pars_for_optimizer = list(data_mod.parameters())

    return SGD(pars_for_optimizer, ell_func, n_samples = nsamples, lr=1e-2)
