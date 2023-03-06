import logging
from pathlib import Path
from typing import Union, Sequence, Tuple, Mapping, Callable
import torch
from torch import Tensor
import gpytorch
from gpytorch.models import ExactGP
import numpy as np
from matplotlib.lines import Line2D
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from spectralgp.models import SpectralModel
from spectralgp.samplers import AlternatingSampler

from models.gaussian_process.models import GPModelCollection
from bayes_quad.surrogates import GPModel
from bayes_quad.quadrature.inter_domain import InterDomainIntegrandModel

# plt.rc('text', usetex=True)
column_width_pt = 240
plt.rc('font', family='serif')
figure_aspect_ratio = 4 / 3
figure_width_fraction = 1
column_width_inches = column_width_pt / 72.27
figure_width = figure_width_fraction * column_width_inches
plt.rc(
    'figure', figsize=(figure_width, figure_width / figure_aspect_ratio)
)
plt.rcParams.update({'font.size': 7})

COLOURS = {
    'purple': '#8159a4',
    'cyan': '#60c4bf',
    'orange': '#f19c39',
    'red': '#cb5763',
    'blue': '#6e8dd7'
}

def wsabi_posterior(
        model_components,
        learn_output: InterDomainIntegrandModel,
        infer_function: Callable,
        train_x: Tensor,
        train_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        numerics: Mapping,
        bounds: Mapping[str, float],
        plot_density: int,
        log_dir: Union[Path, str] = Path('/'),
        feature_preprocessor=None,
        target_preprocessor=None
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    plot_x = torch.linspace(
        bounds['lower'],
        bounds['upper'],
        plot_density
    ).to(train_x)

    mean, variance, _, _, _ = infer_function(
        **model_components,
        learn_output=learn_output,
        test_inputs=plot_x,
        test_targets=None,
        numerics=numerics
    )

    confidence_region = 2 * variance.sqrt()
    lower = (mean - confidence_region).squeeze()
    upper = (mean + confidence_region).squeeze()
    # to numpy
    train_x = train_x.cpu().numpy()
    train_y = train_y.cpu().numpy()
    test_x = test_x.cpu().numpy()
    test_y = test_y.cpu().numpy()
    plot_x = plot_x.cpu().numpy()
    # Initialize plot
    fig, ax = plt.subplots(1, 1)  #, figsize=(12, 4))
    # Plot training and testing data
    ax.plot(
        train_x,
        train_y,
        color=COLOURS['red'],
        marker='.', linestyle=''
    )
    ax.plot(test_x, test_y, color=COLOURS['red'])
    ax.plot(plot_x, mean.detach().cpu().numpy(), COLOURS['blue'])
    ax.fill_between(
        plot_x,
        lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(),
        color=COLOURS['blue'],
        alpha=0.5
    )
    ax.legend(
        ['Train Data', 'Test Data', 'Mean', 'Confidence'], loc='upper right'
    )
    ax.set_xbound(lower=bounds['lower'], upper=bounds['upper'])
    ax.set_ylim(bottom=-3, top=5)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title(f'BQ Moment Matched Posterior')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    fig.tight_layout()

    fig.savefig(log_dir / f'moment_matched_posterior.pdf')
    plt.close(fig)

def fkl_posterior(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        model: SpectralModel,
        sampler: AlternatingSampler,
        bounds: Mapping[str, float] = None,
        plot_density: int = 100,
        n_integration_samples: int = 10,
        log_dir: Union[Path, str] = Path('/')
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    (log_dir / 'posteriors').mkdir(parents=True, exist_ok=True)

    if bounds is None:
        bounds = {}
        x_data = torch.cat((train_x, test_x), dim=0)
        bounds['lower'] = x_data.min().item()
        bounds['upper'] = x_data.max().item()

    plot_x = torch.linspace(
        bounds['lower'],
        bounds['upper'],
        plot_density
    )

    last_samples = min(n_integration_samples, sampler.gsampled[0].size(-1))
    # preprocess the spectral samples #
    out_samples = sampler.gsampled[0][0, :, -last_samples:].detach()

    mu = torch.zeros(last_samples, len(plot_x))
    var = torch.zeros_like(mu)
    lower_pred = torch.zeros_like(mu)
    upper_pred = torch.zeros_like(mu)

    model.eval()
    with torch.no_grad():
        for i in range(last_samples):
            model.covar_module.set_latent_params(out_samples[:, i])
            model.set_train_data(train_x, train_y)
            out = model(plot_x)
            lower_pred[i, :], upper_pred[i, :] = out.confidence_region()
            mu[i, :] = out.mean
            var[i, :] = out.variance

    # Initialize plot
    fig_mm, ax_mm = plt.subplots(1, 1)  # , figsize=(16, 12))

    # Plot training and testing data
    ax_mm.plot(
        train_x.cpu().numpy(), train_y.cpu().numpy(), color=COLOURS['red'], marker='.', linestyle=''
    )
    ax_mm.plot(test_x.cpu().numpy(), test_y.cpu().numpy(), color=COLOURS['red'])

    weights = torch.ones(last_samples)
    weights /= weights.sum()
    weights.unsqueeze_(0)
    # Moment matching
    matched_mu = weights @ mu
    matched_var = weights @ var + weights @ mu ** 2 - (matched_mu) ** 2
    matched_stddev_2 = 2 * matched_var.sqrt()
    lower = (matched_mu - matched_stddev_2).squeeze()
    upper = (matched_mu + matched_stddev_2).squeeze()

    # Plot moment matched mean
    ax_mm.plot(plot_x.cpu().numpy(), matched_mu.squeeze().cpu().numpy(), COLOURS['blue'])
    # Shade between the lower and upper confidence bounds
    ax_mm.fill_between(
        plot_x.cpu().numpy(),
        lower.cpu().numpy(),
        upper.cpu().numpy(),
        color=COLOURS['blue'],
        alpha=0.5
    )
    ax_mm.legend(
        ['Train Data', 'Test Data', 'Mean', 'Confidence'], loc='upper left'
    )
    ax_mm.set_xbound(lower=bounds['lower'], upper=bounds['upper'])
    ax_mm.set_ylim(bottom=-3, top=5)
    # ax_mm.set_xlabel('x')
    # ax_mm.set_ylabel('y')
    ax_mm.axes.xaxis.set_visible(False)
    ax_mm.axes.yaxis.set_visible(False)
    # ax_mm.set_title('FKL Moment Matched Posterior')
    fig_mm.tight_layout()

    fig_mm.savefig(log_dir / 'moment_matched_posterior.pdf')
    plt.close(fig_mm)


def plain_posterior(
        model: ExactGP,
        train_inputs: Tensor = None,
        train_targets: Tensor = None,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        bounds: Mapping[str, float] = None,
        plot_density: int = 100,
        log_dir: Union[Path, str] = Path('./'),
):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    plot_x = torch.linspace(
        bounds['lower'],
        bounds['upper'],
        plot_density
    ).to(train_inputs)

    model.eval()
    model.likelihood.eval()

    posterior = model.likelihood(model(plot_x))
    mean = posterior.mean
    lower, upper = posterior.confidence_region()
    # to numpy
    train_inputs = train_inputs.cpu().numpy()
    train_targets = train_targets.cpu().numpy()
    test_inputs = test_inputs.cpu().numpy()
    test_targets = test_targets.cpu().numpy()
    plot_x = plot_x.cpu().numpy()
    mean = mean.detach().cpu().numpy()
    lower = lower.detach().cpu().numpy()
    upper = upper.detach().cpu().numpy()
    # Initialize plot
    fig, ax = plt.subplots(1, 1)  # , figsize=(16, 12))
    # Plot training and testing data
    ax.plot(
        train_inputs,
        train_targets,
        color=COLOURS['red'],
        marker='.', linestyle=''
    )
    ax.plot(test_inputs, test_targets, color=COLOURS['red'])
    ax.plot(plot_x, mean, COLOURS['blue'])
    ax.fill_between(
        plot_x,
        lower,
        upper,
        color=COLOURS['blue'],
        alpha=0.5
    )
    ax.legend(['Train Data', 'Test Data', 'Mean', 'Confidence'], loc='upper right')
    ax.set_xbound(lower=bounds['lower'], upper=bounds['upper'])
    ax.set_ylim(bottom=-3, top=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    fig.tight_layout()

    fig.savefig(log_dir / f'posterior.pdf')
    plt.close(fig)


def sm_kernel(
    model,
    problem_config,
    feature_preprocessor,
    log_dir
):
    from sklearn.mixture import GaussianMixture
    delta = (feature_preprocessor.data_max_ - feature_preprocessor.data_min_).item()
    fig, ax = plt.subplots(1, 1)
    plot_x = np.linspace(0, 75, 500)
    for w, m, s in zip(
        problem_config['mixture_weights'],
        problem_config['mixture_means'],
        problem_config['mixture_scales']
    ):
        true_scales = np.array(s).reshape(-1, 1) * delta
        true_gmm = GaussianMixture(
            n_components=len(true_scales),
            covariance_type='diag'
        )
        true_gmm.weights_ = np.array(w).reshape(-1)
        true_gmm.means_ = np.array(m).reshape(-1, 1) * delta
        true_gmm.covariances_ = true_scales ** 2
        true_gmm.precisions_ = true_scales ** -2
        true_gmm.precisions_cholesky_ = true_scales ** -1
        true_spectrum = np.exp(true_gmm.score_samples(plot_x.reshape(-1, 1)).reshape(-1))
        ax.plot(plot_x, true_spectrum, color=COLOURS['red'])

    weights = model.covar_module.mixture_weights.squeeze()
    means = model.covar_module.mixture_means.squeeze()
    scales = model.covar_module.mixture_scales.squeeze()
    gmm = GaussianMixture(
        n_components=len(weights),
        covariance_type='diag'
    )
    s_np = scales.detach().cpu().numpy().reshape(-1, 1)
    gmm.weights_ = weights.detach().cpu().numpy()
    gmm.means_ = means.detach().cpu().numpy().reshape(-1, 1)
    gmm.covariances_ = s_np ** 2
    gmm.precisions_ = s_np ** -2
    gmm.precisions_cholesky_ = s_np ** -1
    spectrum = np.exp(gmm.score_samples(plot_x.reshape(-1, 1)).reshape(-1))
    ax.plot(plot_x, spectrum, color=COLOURS['blue'])
    
    legend_elements = [
        Line2D([0], [0], color=COLOURS['red'], label='True Kernels'),
        Line2D([0], [0], color=COLOURS['blue'], label='Learnt Kernel')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xbound(lower=0, upper=30)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    fig.tight_layout()
    fig.savefig(log_dir / 'kernel.pdf')
    plt.close(fig)


def wsabi_kernel(
    learn_output,
    problem_config,
    feature_preprocessor,
    log_dir,
    model: ExactGP,
    train_inputs: Tensor = None,
    train_targets: Tensor = None,
    test_inputs: Tensor = None,
    test_targets: Tensor = None,
    bounds: Mapping[str, float] = None,
    plot_density: int = 100,
):
    from sklearn.mixture import GaussianMixture
    delta = (feature_preprocessor.data_max_ - feature_preprocessor.data_min_).item()
    fig, ax = plt.subplots(1, 1)
    plot_x = np.linspace(0, 75, 500)
    for w, m, s in zip(
        problem_config['mixture_weights'],
        problem_config['mixture_means'],
        problem_config['mixture_scales']
    ):
        true_scales = np.array(s).reshape(-1, 1) * delta
        true_gmm = GaussianMixture(
            n_components=len(true_scales),
            covariance_type='diag'
        )
        true_gmm.weights_ = np.array(w).reshape(-1)
        true_gmm.means_ = np.array(m).reshape(-1, 1) * delta
        true_gmm.covariances_ = true_scales ** 2
        true_gmm.precisions_ = true_scales ** -2
        true_gmm.precisions_cholesky_ = true_scales ** -1
        true_spectrum = np.exp(true_gmm.score_samples(plot_x.reshape(-1, 1)).reshape(-1))
        ax.plot(plot_x, true_spectrum, color=COLOURS['red'])

    bq_weights = learn_output.prediction_weights.sum(0).clamp_(1e-3, 0.999).cpu().numpy().tolist()
    bq_weights_iter = iter(bq_weights)
    
    for h in learn_output.prediction_hyperparameters:
        num_components = int(h.shape[1] / 3)
        weights = h[:, :num_components]
        means = h[:, num_components:-num_components]
        scales = h[:, -num_components:]
        for (w, m, s) in zip(weights, means, scales):
            gmm = GaussianMixture(
                n_components=num_components,
                covariance_type='diag'
            )
            s_np = s.cpu().numpy().reshape(-1, 1)
            gmm.weights_ = w.cpu().numpy()
            gmm.means_ = m.cpu().numpy().reshape(-1, 1)
            gmm.covariances_ = s_np ** 2
            gmm.precisions_ = s_np ** -2
            gmm.precisions_cholesky_ = s_np ** -1
            spectrum = np.exp(gmm.score_samples(plot_x.reshape(-1, 1)).reshape(-1))
            ax.plot(plot_x, spectrum, color=COLOURS['blue'], alpha=next(bq_weights_iter))
    
    legend_elements = [
        Line2D([0], [0], color=COLOURS['red'], label='True Kernels'),
        Line2D([0], [0], color=COLOURS['blue'], label='Sampled Kernels')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xbound(lower=0, upper=30)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    fig.tight_layout()
    fig.savefig(log_dir / 'kernel.pdf')
    plt.close(fig)

    def extract_best_hyperparameters(bqw, prediction_hyperparameters):
        max_index = np.argmax(bqw)
        for i, i_hyps in enumerate(prediction_hyperparameters):
            for hyps in i_hyps:
                max_index -= 1
                if max_index == -1:
                    return i, hyps
    model_index, hyps = extract_best_hyperparameters(bq_weights, learn_output.prediction_hyperparameters)
    best_model = model.create_model(model_index, hyps)['model']
    plain_posterior(
        best_model,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        bounds,
        plot_density,
        log_dir
    )


def visualise(
        model_components,
        learn_output=None,
        infer_function: Callable = None,
        train_inputs: Tensor = None,
        train_targets: Tensor = None,
        test_inputs: Tensor = None,
        test_targets: Tensor = None,
        feature_preprocessor=None,
        target_preprocessor=None,
        numerics: Mapping = None,
        problem_config: Mapping = None,
        bounds: Mapping[str, float] = None,
        plot_density: int = 250,
        log_dir: Union[Path, str] = Path('/'),
        **kwargs
):
    logging.info('Plotting')
    if isinstance(model_components['model'], SpectralModel):
        fkl_posterior(
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            model_components['model'],
            learn_output,
            bounds=bounds,
            plot_density=plot_density,
            log_dir=log_dir
        )
    elif isinstance(model_components['model'], GPModelCollection):
        wsabi_posterior(
            model_components,
            learn_output,
            infer_function,
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            numerics,
            bounds,
            plot_density,
            log_dir,
            feature_preprocessor,
            target_preprocessor
        )
        wsabi_kernel(
            learn_output=learn_output,
            problem_config=problem_config,
            feature_preprocessor=feature_preprocessor,
            log_dir=log_dir,
            model=model_components['model'],
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            bounds=bounds,
            plot_density=plot_density
        )
    else:
        plain_posterior(
            model_components['model'],
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            bounds,
            plot_density,
            log_dir
        )
        sm_kernel(
            model_components['model'],
            problem_config=problem_config,
            feature_preprocessor=feature_preprocessor,
            log_dir=log_dir,
        )
