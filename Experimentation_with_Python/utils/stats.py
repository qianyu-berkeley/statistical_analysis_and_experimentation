# -*- coding: utf-8 -*-
# @Author: qianyu-berkeley

from __future__ import division
import numpy as np
import scipy.stats as scs


def z_val(sig_level=0.05, two_tailed=True):
    """Returns the z value for a given significance level

    Parameters
    ----------
    sig_level: significant level
    two_tailed: if True, 2 tailed; if False, 1 tailed

    Returns:
    ---------
    float
    """
    z_dist = scs.norm()
    if two_tailed:
        sig_level = sig_level/2
        area = 1 - sig_level
    else:
        area = 1 - sig_level

    return z_dist.ppf(area)


def t_welch_bin(N_A, N_B, p_A, p_B, tails=2):
    """Welch's t-test for two unequal-size samples of binormal distribution, not
    assuming equal variances. return t statistics and p value

    Parameters:
    ---------
    N_A: sample size group A
    N_B: sample size group B
    p_A: conversion samples A
    p_B: conversion samples B
    tails: num of tails (1 | 2)

    Returns:
    ---------
    float, float
    """
    assert tails in (1, 2), "invalid: tails must be 1 or 2, found %s" \
        % str(tails)

    if N_A < 10 or N_B < 10:
        print('sample size is too small (< 10)')
        return
    else:
        if p_B <= p_A:
            t_obs = 0
            p_value = 0
        else:
            V_A = p_A*(1-p_A)
            V_B = p_B*(1-p_B)

            # Welch-Satterthwaite equation for degree of freedom
            df = int((V_A/N_A + V_B/N_B)**2 /
                     ((V_A/N_A)**2 / (N_A - 1) + (V_B/N_B)**2 / (N_B - 1)))
            t_obs = (p_A - p_B) / np.sqrt(V_A/N_A + V_B/N_B)
            p_value = tails * scs.t.sf(abs(t_obs), df)
    return t_obs, p_value


def t_welch_norm(x, y, tails=2):
    """Welch's t-test for two unequal-size samples normal distribution, not
    assuming equal variances. return t statistics and p value

    Parameters:
    -----------
    x: samples x
    y: samples y
    tails: num of tails

    Returns:
    ---------
    float, float
    """
    assert tails in (1, 2), "invalid: tails must be 1 or 2, found %s" \
        % str(tails)
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = x.size, y.size

    if nx < 10 or ny < 10:
        print('sample size is too small (< 10)')
        return
    else:
        vx, vy = x.var(), y.var()

        # Welch-Satterthwaite equation for degree of freedom
        df = int((vx/nx + vy/ny)**2 /
                 ((vx/nx)**2 / (nx - 1) + (vy/ny)**2 / (ny - 1)))
        t_obs = (x.mean() - y.mean()) / np.sqrt(vx/nx + vy/ny)
        p_value = tails * scs.t.sf(abs(t_obs), df)
    return t_obs, p_value


def confidence_interval(mean=0, se=1, sig_level=0.05):
    """Returns the confidence interval as a tuple

    Parameters
    ----------
    mean: sample mean
    se: standard error (sdv/sqrt(sample_size))
    sig_level: significant level

    Returns:
    ---------
    tuple
    """
    z = z_val(sig_level)
    left = mean - z * se
    right = mean + z * se
    return (left, right)


def pooled_prob(N_A, N_B, X_A, X_B):
    """Returns pooled probability for two (A/B) samples

    Parameters
    ----------
    N_A: sample size group A
    N_B: sample size group B
    X_A: prob_A * N_A
    X_B: prob_B * N_B

    Returns:
    ---------
    float
    """
    return (X_A + X_B) / (N_A + N_B)


def pooled_SE(N_A, N_B, X_A, X_B):
    """Returns the pooled standard error for two (A/B) samples

    Parameters
    ----------
    N_A: sample size group A
    N_B: sample size group B
    X_A: prob_A * N_A
    X_B: prob_B * N_B

    Returns:
    ---------
    float
    """
    p_hat = pooled_prob(N_A, N_B, X_A, X_B)
    se = np.sqrt(p_hat * (1 - p_hat) * (1 / N_A + 1 / N_B))
    return se


def alpha(mean=0, se=1, sig_level=0.05, side='right'):
    """Returns the critical value of (alphas) of one-tailed normal distribution


    Parameters
    ----------
    mean: sample mean
    sdv: sample standard deviation
    size: sample size
    sig_level: significant level
    side: right or left

    Returns:
    ---------
    float
    """
    z = z_val(sig_level, two_tailed=False)
    if side == 'right':
        alpha = mean + z * se
    else:
        alpha = mean - z * se
    return alpha


def p_val_Z(N_A, N_B, p_A, p_B, two_tailed=True):
    """Returns the p-value for an A/B test using z score and caclulated with
       survival function (scipy.stats.norm.sf(): sf = 1 - cdf)

    Parameters
    ----------
    N_A: sample size group A
    N_B: sample size group B
    p_A: conversion samples A
    p_B: conversion samples B
    two_tailed: if True, 2 tailed; if False, 1 tailed

    Return:
    ----------
    float
    """
    SE_A = np.sqrt(p_A * (1 - p_A) / N_A)
    SE_B = np.sqrt(p_B * (1 - p_B) / N_B)
    Z = (p_B - p_A) / np.sqrt(SE_A**2 + SE_B**2)

    if two_tailed:
        p_value = scs.norm.sf(abs(Z))*2
    else:
        p_value = scs.norm.sf(abs(Z))
    return p_value


def power(N_A, N_B, p_A, p_B, sig_level=0.05):
    """Return statistical power of A/B Test

    Parameters
    ----------
    N_A: sample size group A
    N_B: sample size group B
    p_A: conversion samples A
    p_B: conversion samples B
    sig_level: significant level

    Return:
    ----------
    float
    """
    bcr = p_A  # Base
    d_hat = p_B - p_A  # Effect size

    if d_hat <= 0:
        return 0.0
    else:
        X_A = bcr * N_A
        X_B = (bcr + d_hat) * N_B
        se = pooled_SE(N_A, N_B, X_A, X_B)
        alternative = scs.norm(d_hat, se)
        right = alpha(mean=0, se=se, sig_level=sig_level)
        power = 1 - alternative.cdf(right)
        return power


def min_sample_size(bcr, mde, power=0.8, sig_level=0.05):
    """Returns the minimum sample size to set up a split test

    Parameters:
    -----------
    bcr (float): conversion for the control aka baseline conversion rate
    mde (float): minimum change in conversion between control group and
    test group if alternative hypothesis is true aka minimum effect size
    power (float): probability of rejecting the null hypothesis when the
    null hypothesis is false, default at 0.8
    sig_level (float): significance level (alpha) default at 0.05

    Returns:
    --------
        min_N: minimum sample size (float)
    """
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)

    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)

    # find Z_alpha
    Z_alpha = standard_norm.ppf(1-sig_level/2)

    # average of probabilities from both groups
    pooled_prob = (bcr + bcr+mde) / 2

    min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
             / mde**2)
    return min_N
