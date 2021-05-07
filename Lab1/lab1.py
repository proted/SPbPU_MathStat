import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


def get_sample_range(sample):
    a, b = math.floor(min(sample)), math.ceil(max(sample))
    while b - a > 20:
        b /= 2
        a /= 2
    return a, b


def get_sample_partition(sample, step):
    left, right = get_sample_range(sample)
    n = math.ceil((right - left) / step)
    return [left + i * step for i in range(n + 1)]


def get_continuous_distribution_sample_plot(sample_size: int,
                                            distribution: stats.rv_continuous,
                                            distribution_params: tuple):
    sample = distribution.rvs(*distribution_params, size=sample_size)
    probability_density_points = np.arange(*get_sample_range(sample), 0.05)
    probability_density_values = distribution.pdf(probability_density_points, *distribution_params)
    return sample, probability_density_points, probability_density_values


def get_poisson_distribution_sample_plot(sample_size: int,
                                         distribution_params: tuple):
    sample = stats.poisson.rvs(*distribution_params, size=sample_size)
    probability_density_points = np.arange(*get_sample_range(sample), 1)
    probability_density_values = stats.poisson.pmf(probability_density_points, *distribution_params)
    return sample, probability_density_points, probability_density_values


def do_research():
    sample_sizes = [10, 50, 1000]
    continuous_distributions = [(stats.norm, (0, 1), 'Normal distribution'),
                                (stats.laplace, (0, 1 / math.sqrt(2)), 'Laplace distribution'),
                                (stats.cauchy, (0, 1), 'Cauchy distribution'),
                                (stats.uniform, (-math.sqrt(3), 2*math.sqrt(3)), 'Uniform distribution')]

    for continuous_distribution in continuous_distributions:
        for sample_size in sample_sizes:
            fig, ax = plt.subplots(1, 1)
            (sample,
             probability_density_points,
             probability_density_values) = get_continuous_distribution_sample_plot(sample_size,
                                                                                   continuous_distribution[0],
                                                                                   continuous_distribution[1])
            ax.hist(sample, bins=get_sample_partition(sample, 0.5), density=True, rwidth=0.95, range=get_sample_range(sample))
            ax.plot(probability_density_points, probability_density_values)
            ax.set_title(continuous_distribution[2] + ', sample size = ' + str(sample_size))
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            plt.savefig('Lab1/pictures/' + continuous_distribution[2] + '_' + str(sample_size))
    for sample_size in sample_sizes:
        fig, ax = plt.subplots(1, 1)
        (sample,
         probability_density_points,
         probability_density_values) = get_poisson_distribution_sample_plot(sample_size, (10, 0))
        ax.hist(sample, bins=get_sample_partition(sample, 1), density=True, rwidth=0.95, range=get_sample_range(sample))
        ax.set_title('Poisson distribution' + 'sample size = ' + str(sample_size))
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.plot(probability_density_points, probability_density_values)
        plt.savefig('Lab1/pictures/Poisson distribution_' + str(sample_size))
