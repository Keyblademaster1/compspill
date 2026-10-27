"""Small robust-statistics helpers (numpy only)."""
from __future__ import annotations

import numpy as np


def robust_slope(x, y, n_iter=50, c=1.345):
    """Huber-weighted (IRLS) fit of y = a + b*x.  Returns (a, b).

    Plain least squares is pulled around by the few odd events every control has
    (doublets, debris, autofluorescent cells); Huber weights down-weight events
    whose residual is more than c robust SDs, so the slope follows the bulk.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    w = np.ones_like(x)
    beta = np.zeros(2)
    for _ in range(n_iter):
        Wx = X * w[:, None]
        new = np.linalg.lstsq(Wx.T @ X, Wx.T @ y, rcond=None)[0]
        r = y - X @ new
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / u)
        if np.allclose(new, beta, rtol=1e-7, atol=1e-10):
            break
        beta = new
    return beta


def robust_sd(v):
    """Half the 15.87-84.13 percentile range (= SD for a normal distribution)."""
    lo, hi = np.percentile(v, [15.87, 84.13])
    return (hi - lo) / 2


def mad(v, axis=None):
    med = np.median(v, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(v - med), axis=axis)


def otsu_threshold(v, bins=256):
    """Threshold that best splits v into two groups (Otsu's method on a histogram)."""
    h, edges = np.histogram(v, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * mids) / np.maximum(w0, 1)
    m1 = (np.sum(h * mids) - np.cumsum(h * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return mids[np.argmax(between)]


def two_gaussians(v, n_iter=300):
    """1-D two-component Gaussian mixture by EM.

    Returns dict with means, sds, weights, log-likelihoods of the 1- and
    2-component fits, BIC difference (positive = two components fit better),
    Ashman's D (separation of the components; > 2 means clearly separate) and the
    density valley between the two means (a natural threshold).
    """
    v = np.asarray(v, float)
    n = len(v)
    mu1, sd1 = v.mean(), v.std() + 1e-12
    ll1 = np.sum(-0.5 * ((v - mu1) / sd1) ** 2 - np.log(sd1 * np.sqrt(2 * np.pi)))
    med = np.median(v)
    lo, hi = v[v <= med], v[v > med]
    if len(lo) < 2 or len(hi) < 2:
        return dict(means=(mu1, mu1), sds=(sd1, sd1), weights=(1., 0.), ll1=ll1, ll2=ll1,
                    delta_bic=0., ashman_d=0., valley=mu1)
    mu = np.array([lo.mean(), hi.mean()])
    sd = np.array([lo.std(), hi.std()]) + 1e-6 * sd1
    w = np.array([.5, .5])
    ll2 = -np.inf
    for _ in range(n_iter):
        logp = -0.5 * ((v[:, None] - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi)) + np.log(w)
        mx = logp.max(1, keepdims=True)
        new_ll = np.sum(mx[:, 0] + np.log(np.exp(logp - mx).sum(1)))
        r = np.exp(logp - mx)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-12
        w = nk / n
        mu = (r * v[:, None]).sum(0) / nk
        sd = np.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / nk) + 1e-6 * sd1
        if abs(new_ll - ll2) < 1e-8 * abs(new_ll):
            ll2 = new_ll
            break
        ll2 = new_ll
    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]
    grid = np.linspace(mu[0], mu[1], 200)
    dens = (w / sd * np.exp(-0.5 * ((grid[:, None] - mu) / sd) ** 2)).sum(1)
    return dict(means=tuple(mu), sds=tuple(sd), weights=tuple(w), ll1=ll1, ll2=ll2,
                delta_bic=(-2 * ll1 + 2 * np.log(n)) - (-2 * ll2 + 5 * np.log(n)),
                ashman_d=float(np.sqrt(2) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)),
                valley=float(grid[np.argmin(dens)]))
