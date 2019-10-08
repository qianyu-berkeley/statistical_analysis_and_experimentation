# -*- coding: utf-8 -*-
# @Author: qianyu-berkeley
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
# Local stats modules
from .stats import pooled_SE, confidence_interval, p_val_Z, z_val, alpha,\
    t_welch_bin, power
plt.style.use('seaborn')


def plot_CI(ax, mu, se, sig_level=0.05, color='blue', two_tailed=True):
    """plot the confidence interval

    Parameters:
    -----------
    ax: matplotlib axes)
    mu: sample mean
    sdv: standard deviation
    sig_level: significant level
    color: line color
    two_tailed: whether it is 2-tailed or 1-tailed

    Returns:
    --------
    """
    left, right = confidence_interval(mean=mu, se=se, sig_level=sig_level)
    ax.axvline(left, c=color, linestyle='--', alpha=0.5)
    ax.axvline(right, c=color, linestyle='--', alpha=0.5)


def plot_norm_dist(ax, mu, sdv, with_CI=False, sig_level=0.05, label=None):
    """Plot a normal distribution to the axes provided

    Parameters:
    -----------
    ax: matplotlib axes
    mu: mean of the normal distribution
    sdv: standard deviation of the normal distribution
    with_CI: add confidence interval line
    sig_level: Significant level

    Returns:
    --------
    """
    x = np.linspace(mu - 15 * sdv, mu + 15 * sdv, 1000)
    y = scs.norm(mu, sdv).pdf(x)
    ax.plot(x, y, label=label)
    if with_CI:
        plot_CI(ax, mu, sdv, sig_level=sig_level)


def plot_binom_dist(ax, N, p, label=None):
    """plot a binomial distribution to the axes provided

    Example:
    --------
    plot_binom_dist(ax, 0, 1) plots a binormal distribution

    Parameters:
    -----------
    ax: matplotlib axes
    N: number of samples
    p: conversion rate (probability of conversion)
    label: plot label

    Returns:
    --------
    """
    x = np.linspace(0, N, N+1)
    y = scs.binom(N, p).pmf(x)
    ax.plot(x, y, label=label)


def plot_alpha(ax, mu, se, sig_level=0.05, color='red'):
    """Plot the one-tailed critical value (alpha)

    Parameters:
    -----------
    ax: matplotlib axes
    mu: mean
    sdv: standard deviation
    sig_level: significant level

    Returns:
    --------
    """

    right = alpha(mean=mu, se=se, sig_level=sig_level)
    ax.axvline(right, c=color, linestyle='--', alpha=0.5)


def plot_null(ax, se, sig_level=0.05, two_tailed=True):
    """Plots the null hypothesis distribution: normally
    distributed with mean at 0 with standard deviation at pooled standard error
    and plot confidence interval band if 2 tailed, plot alpha if 1 tailed

    Parameters:
    ----------
    ax: matplotlib axes
    se: the pooled standard error of the control and test group
    sig_level: significant level
    two_tailed: if true 2 tailed, if false 1 tailed

    Returns:
    --------
    """
    plot_norm_dist(ax, 0, se, label="Null hypothesis")
    if two_tailed:
        plot_CI(ax, mu=0, se=se, color='black', sig_level=sig_level)
    else:
        plot_alpha(ax, mu=0, se=se, sig_level=sig_level)


def plot_alter(ax, se, d_hat, sig_level=0.05):
    """Plots the alternative hypothesis distribution: normally distributed
    with mean at d_hat with standard deviation of pooled standard error
    Plot confidence interval band.

    Parameters:
    -----------
    ax: matplotlib axes
    se: the pooled standard error of the control and test group
    d_hat: effect size
    sig_level: significant level

    Returns:
    --------
    """
    plot_norm_dist(ax, d_hat, se, label="Alternative hypothesis")
    plot_CI(ax, mu=d_hat, se=se, color='purple', sig_level=sig_level)


def ab_dist(se, d_hat=0, group_type='control'):
    """Returns a normal distribution object

    Parameters:
    ----------
    se: pooled standard error of two independent samples
    d_hat: effect size
    group: 'control' or 'test'

    Returns:
    --------
    scipy.stats object
    """
    if group_type == 'control':
        return scs.norm(0, se)
    elif group_type == 'test':
        return scs.norm(d_hat, se)


def abplot(N_A, N_B, bcr, d_hat, sig_level=0.05, plot_power=False,
           plot_alpha=False, plot_beta=False, add_p_value=False,
           legend=True, two_tailed=True):
    """Plot A/B test results

    Parameters:
    -----------
    N_A: sample size of group A
    N_B: sample size of group B
    bcr (float): base conversion rate i.e. conversion rate of control
    d_hat: effect size
    sig_level: significant level
    plot_power: add power to the plot
    plot_alpha: add alpha to the plot
    plot_beta: add beta to the plot
    add_p_value: show p_value
    legend: show legend
    two_tailed: if True 2 tailed, if False 1 tailed

    Returns:
    --------
    """
    # create a plot object
    fig, ax = plt.subplots(figsize=(12, 6))

    # define parameters to find pooled standard error
    X_A = bcr * N_A
    X_B = (bcr + d_hat) * N_B
    se = pooled_SE(N_A, N_B, X_A, X_B)

    # plot the distribution of the null and alternative hypothesis
    plot_null(ax, se, sig_level=sig_level, two_tailed=two_tailed)
    plot_alter(ax, se, d_hat, sig_level=sig_level)

    # set extent of plot area
    ax.set_xlim(-3 * d_hat, 3 * d_hat)

    # shade areas according to user input
    if plot_power:
        fill_area(ax, d_hat, se, sig_level,
                  area_type='power', two_tailed=False)
    if plot_alpha:
        fill_area(ax, d_hat, se, sig_level,
                  area_type='alpha', two_tailed=False)
    if plot_beta:
        fill_area(ax, d_hat, se, sig_level,
                  area_type='beta', two_tailed=False)

    # show p_value based on the binomial distributions for the two groups
    if add_p_value:
        null = ab_dist(se, 'control')
        _, pval_t = t_welch_bin(N_A, N_B, bcr, bcr+d_hat, tails=1)
        pval_z = p_val_Z(N_A, N_B, bcr, bcr+d_hat, two_tailed=False)

        if N_A < 30 or N_B < 30:
            ax.text(6 * se, null.pdf(0),
                    'p-value T-test = {:0.3f}'.format(pval_t),
                    fontsize=12, ha='left')
        else:
            ax.text(6 * se, null.pdf(0),
                    'p-value Z-test = {:0.3f}'.format(pval_z),
                    fontsize=12, ha='left')

    if legend:
        plt.legend(loc='upper left')

    plt.xlabel("Cohen's d")
    plt.ylabel('Probability Distribution')
    plt.show()


def fill_area(ax, d_hat, se, sig_level, area_type='power',
              two_tailed=True):
    """Fill area of the A/B test distributions based on type

    Parameters:
    -----------
    ax: matplotlib axies
    d_hat: effect size
    se: pooled standard error of the control and the test groups
    sig_level: significant level
    area_type: fill area type, default is power
    two_tailed: if True 2 tailed, if False 1 tailed

    Returns:
    --------
    """
    if two_tailed:
        left, right = confidence_interval(mean=0, se=se,
                                          sig_level=sig_level)
    else:
        right = alpha(mean=0, se=se, sig_level=sig_level)

    x = np.linspace(-15 * se, 15 * se, 1000)
    null = ab_dist(se, 'control')
    alternative = ab_dist(se, d_hat, 'test')

    # alpha
    # the upper significance boundary of the null distribution
    if area_type == 'alpha':
        ax.fill_between(x, 0, null.pdf(x), color='firebrick', alpha='0.25',
                        where=(x > right))
        ax.text(-3 * se, null.pdf(0),
                'alpha = {0:.3f}'.format(1 - null.cdf(right)),
                fontsize=10, ha='right', color='k')

    # beta
    # between alternative hypothesis and the upper significance boundary
    # of null hypothesis
    if area_type == 'beta':
        ax.fill_between(x, 0, alternative.pdf(x), color='navy', alpha='0.25',
                        where=(x < right))
        ax.text(0 * se, null.pdf(0),
                'beta = {0:.3f}'.format(alternative.cdf(right)),
                fontsize=10, ha='right', color='k')

    # power
    # between upper significance boundary of the null and
    # alternative distribution
    if area_type == 'power':
        ax.fill_between(x, 0, alternative.pdf(x), color='aqua', alpha='0.25',
                        where=(x > right))
        ax.text(3 * se, null.pdf(0),
                'power = {0:.3f}'.format(1 - alternative.cdf(right)),
                fontsize=10, ha='right', color='k')


def plot_CI_multi_OvAll(N, X, sig_level=0.05, dmin=None):
    """Returns a confidence interval bar plot for multivariate tests
    one control vs. multiple test

    Parameters:
    ------------
    N: sample size list or tuple for each variant (assume control is the first
    element)
    X: conversions list or tuple for each variant (assume control is the first
    element)
    sig_level: significance level
    dmin: minimum effective size

    Returns:
    ----------
    """

    # initiate plot object
    fig, ax = plt.subplots(figsize=(12, 3))

    # get control group values
    N_A = N[0]
    X_A = X[0]

    # initiate containers for standard error and differences
    SE = []
    d = []

    # iterate through X and N and calculate d and SE
    for idx in range(1, len(N)):
        X_B = X[idx]
        N_B = N[idx]
        d.append(X_B / N_B - X_A / N_A)
        SE.append(pooled_SE(N_A, N_B, X_A, X_B))
    SE = np.array(SE)
    d = np.array(d)

    # z value
    z = z_val(sig_level)
    ci = SE * z

    # bars to represent the confidence interval
    y = np.arange(len(N)-1)
    ax.hlines(y, d-ci, d+ci, color='blue', alpha=0.4, lw=10, zorder=1)
    # marker for the mean
    ax.scatter(d, y, s=300, marker='|', lw=10, color='magenta', zorder=2)

    # vertical line to represent 0
    ax.axvline(0, c='grey', linestyle='-')

    # plot dmin
    if dmin is not None:
        ax.axvline(-dmin, c='red', linestyle='--', alpha=0.75)
        ax.axvline(dmin, c='green', linestyle='--', alpha=0.75)

    # invert y axis to show variant 1 at the top
    ax.invert_yaxis()

    # label variants on y axis
    labels = ['variant{}'.format(i+1) for i in range(len(N)-1)]
    plt.yticks(np.arange(len(N)-1), labels)


def plot_CI_multi_pairs(A, B, sig_level=0.05):
    """Returns a confidence interval bar for multivariate tests
    multiple control vs test pairs (e.g. customer funnel)

    Parameters:
    ------------
    A: (sample size, conversion) pair list or tuples for control groups
    B: (sample size, conversion) pair list or tuples for test group
    sig_level: significance level

    Returns:
    ----------
    """

    # initiate plot object
    fig, ax = plt.subplots(figsize=(12, 3))

    # initiate containers for standard error and differences
    SE = []
    d = []
    # iterate through X and N and calculate d and SE
    for i in range(len(A)):
        X_A = A[i][1]
        N_A = A[i][0]
        X_B = B[i][1]
        N_B = B[i][0]
        d.append(X_B / N_B - X_A / N_A)
        SE.append(pooled_SE(N_A, N_B, X_A, X_B))

    # convert to numpy arrays
    SE = np.array(SE)
    d = np.array(d)

    # z value
    z = z_val(sig_level)

    # confidence interval values
    ci = SE * z

    # bar to represent the confidence interval
    y = np.arange(len(A))
    ax.hlines(y, d-ci, d+ci, color='blue', alpha=0.4, lw=10, zorder=1)
    # marker for the mean
    ax.scatter(d, y, s=300, marker='|', lw=10, color='magenta', zorder=2)

    # vertical line to represent 0
    ax.axvline(0, c='grey', linestyle='-')

    # invert y axis to show variant 1 at the top
    ax.invert_yaxis()

    # label variants on y axis
    labels = ['metric{}'.format(i+1) for i in range(len(A))]
    plt.yticks(np.arange(len(A)), labels)


def zplot(conf=0.95, two_tailed=True, one_tail_right=False):
    """Plots a z distribution based on statistics property

    Parameters:
    ----------
    conf: The area under the normal distribution
    two_tailed: if true 2 tailed if false 1 tailed
    one_tail_right: if 1 tailed, left side or right side, default is left

    Returns:
    ----------
    """
    # create plot object
    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()

    # plot normal distribution
    norm = scs.norm()
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)
    ax.plot(x, y)

    # for two-tailed tests
    if two_tailed:
        left = norm.ppf(0.5 - conf / 2)
        right = norm.ppf(0.5 + conf / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
        ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                        where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                 rotation=90, va="bottom", ha="right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                 fontsize=12, rotation=90, va="bottom", ha="left")
    else:
        # align the conf to the right
        if one_tail_right:
            left = norm.ppf(1-conf)
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                     fontsize=12, rotation=90, va="bottom", ha="right")
        else:
            right = norm.ppf(conf)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                     fontsize=12, rotation=90, va="bottom", ha="left")

    plt.text(0, 0.1, "shaded area = {0:.3f}".format(conf), fontsize=12,
             ha='center')

    # axis labels
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.show()
