#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import scipy.stats
import numpy as np

class Intensity(object):
    __metaclass__ = abc.ABCMeta

class IntensityHomogenuosPoisson(Intensity):

    def __init__(self, lam):
        self.lam = lam

    def get_value(self, t=None, past_ts=None):
        return self.lam

    def get_upper_bound(self, past_ts=None, t=None, to_t=None):
        return self.lam

class IntensityGaussianMixture(Intensity):

    def __init__(self, k=2, centers=[2, 4], stds=[1, 1], coefs= [1, 1]):
        self.k       = k
        self.centers = centers
        self.stds    = stds
        self.coefs   = coefs

    def get_value(self, t=None, past_ts=None):
        return self._get_gaussianmixture_value(t)

    def get_upper_bound(self, past_ts=None, t=None, to_t=None):
        max_val = sum([ self._get_gaussianmixture_value(center) for center in self.centers ])
        return max_val

    def _get_gaussianmixture_value(self, t):
        inten = 0
        for i in range(self.k):
            inten += self.coefs[i] * scipy.stats.norm.pdf(t, self.centers[i], self.stds[i])
        return inten

    def get_integral(self, t, past_ts=None):
        return sum([ coef * (scipy.stats.norm.cdf(t, center, std) - \
                     scipy.stats.norm.cdf(0, center, std))
                     for coef, center, std in zip(self.coefs, self.centers, self.stds) ])

class IntensityHawkes(Intensity):

    def __init__(self, mu=1, alpha=0.3, beta=1):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta

    def get_value(self, t=None, past_ts=None):
        inten = self.mu + np.sum(self.alpha * self.beta * np.exp(-self.beta * np.subtract(t, past_ts)))
        return inten

    def get_upper_bound(self, past_ts=None, t=None, to_t=None):
        max_val = self.mu + np.sum(self.alpha * self.beta * np.exp(-self.beta * np.subtract(t, past_ts)))
        return max_val

    def get_integral(self, t, past_ts):
        return self.mu * t + \
               self.alpha * np.sum(1 - np.exp(-self.beta * (t - np.array(past_ts))))

class IntensityPoly(Intensity):

    def __init__(self, segs=[0, 1, 2, 3], b=0, A=[1, 2, -3]):
        self.segs = segs
        self.b    = b
        self.A    = A
        if len(A) != len(segs) - 1:
            raise Exception("Inequality lies in the numbers of segs and A.")

    def get_value(self, t=None, past_ts=None):
        return self._get_poly_value(t)

    def get_upper_bound(self, past_ts=None, t=None, to_t=None):
        max_val = 0
        segs_within_range = [ s for s in self.segs if s > t and s < to_t ]
        if len(segs_within_range) > 0:
            max_val = max([ self._get_poly_value(t) for s in segs_within_range ])
        max_val = max([ self._get_poly_value(t), self._get_poly_value(to_t), max_val ])
        return max_val

    def _get_poly_value(self, t):
        if t > self.segs[-1]:
            raise Exception("t is out of range.")
        segs_before_t = [ s for s in self.segs if s < t ]
        b = self.b
        for seg_ind in range(len(segs_before_t)-1):
            b = b + self.A[seg_ind] * (segs_before_t[seg_ind+1] - segs_before_t[seg_ind])
        if len(segs_before_t) >= 1:
            value = b + self.A[len(segs_before_t)-1] * (t - segs_before_t[len(segs_before_t)-1])
        else:
            value = b
        return value

    def get_integral(self, t, past_ts=None):
        if t > self.segs[-1]:
            raise Exception("t is out of range.")
        segs_before_t = [ s for s in self.segs if s < t ]

        # get starting intercepts (bs) for each of segments (size = len(segs_before_t) + 1)
        bs = [self.b]
        for seg_ind in range(len(segs_before_t)-1):
            b = bs[seg_ind] + self.A[seg_ind] * \
                (segs_before_t[seg_ind+1] - segs_before_t[seg_ind])
            bs.append(b)
        bs.append(self._get_poly_value(t)) # last intercept

        # get length of each of segments (size = len(segs_before_t))
        lens = []
        for seg_ind in range(len(segs_before_t)-1):
            lens.append(segs_before_t[seg_ind+1] - segs_before_t[seg_ind])
        last_seg = segs_before_t[-1] if len(segs_before_t) > 0 else 0
        lens.append(t - last_seg) # lengths of last segments

        # get integrals (area) for each of segments
        integrals = [ (width1 + width2) * height / 2.
                      for width1, width2, height in zip(bs[:-1], bs[1:], lens) ]
        return sum(integrals)
#
# class IntensitySelfCorrecting(Intensity):
#
#     def __init__(self, mu=1, alpha=0.3):
#         self.mu    = mu
#         self.alpha = alpha
#
#     def get_value(self, t, past_ts):
#         return np.exp(self.mu * t - self.alpha * len(past_ts))
#
#     def get_upper_bound(self, past_ts=None, t=None, to_t):
#         # TODO: Improve this upper bound
#         return np.exp(self.mu * to_t)
#
#     def get_integral(self, t, past_ts):
#         for past_t in past_ts:
#             past_t
#         return

class IntensityHawkesPlusPoly(IntensityHawkes, IntensityPoly):

    def __init__(self, mu=1, alpha=0.3, beta=1,
                       segs=[0, 1, 2, 3], b=0, A=[1, 2, -3]):
        IntensityPoly.__init__(self, segs=segs, b=b, A=A)
        IntensityHawkes.__init__(self, mu=mu, alpha=alpha, beta=beta)

    def get_value(self, t=None, past_ts=None):
        return IntensityHawkes.get_value(self, t=t, past_ts=past_ts) + \
               IntensityPoly.get_value(self, t=t)

    def get_upper_bound(self, past_ts=None, t=None, to_t=None):
        return IntensityPoly.get_upper_bound(self, t=t, to_t=to_t) + \
               IntensityHawkes.get_upper_bound(self, past_ts=past_ts, t=t)

    def get_integral(self, t, past_ts):
        return IntensityPoly.get_integral(self, t=t) + \
               IntensityHawkes.get_integral(self, t=t, past_ts=past_ts)

class IntensityHawkesPlusGaussianMixture(IntensityHawkes, IntensityGaussianMixture):

    def __init__(self, mu=1, alpha=0.3, beta=1,
                       k=2, centers=[2, 4], stds=[1, 1], coefs=[1, 1]):
        IntensityHawkes.__init__(self, mu=mu, alpha=alpha, beta=beta)
        IntensityGaussianMixture.__init__(self, k=k, centers=centers, stds=stds, coefs=coefs)

    def get_value(self, t=None, past_ts=None):
        return IntensityHawkes.get_value(self, t=t, past_ts=past_ts) + \
               IntensityGaussianMixture.get_value(self, t=t)

    def get_upper_bound(self, past_ts=None, t=None, to_t=None):
        return IntensityGaussianMixture.get_upper_bound(self, t=t, to_t=to_t) + \
               IntensityHawkes.get_upper_bound(self, past_ts=past_ts, t=t)

    def get_integral(self, t, past_ts):
        return IntensityGaussianMixture.get_integral(self, t=t) + \
               IntensityHawkes.get_integral(self, t=t, past_ts=past_ts)

def generate_sample(intensity, T, n):
    seqs = []
    i    = 0
    while True:
        past_ts = []
        cur_t   = 0
        while True:
            intens1 = intensity.get_upper_bound(past_ts=past_ts, t=cur_t, to_t=T)
            intens1 = intens1 if intens1 != 0 else 1e-4
            t_delta = np.random.exponential(1.0/float(intens1))
            next_t  = cur_t + t_delta
            # print "cur_t:%f, next_t:%f, delta_t:%f" % (cur_t, next_t, t_delta)
            if next_t > T:
                break
            intens2 = intensity.get_value(t=next_t, past_ts=past_ts)
            u       = np.random.uniform()
            if float(intens2)/float(intens1) >= u:
                past_ts.append(next_t)
            cur_t = next_t
        if len(past_ts) > 1:
            seqs.append(past_ts)
            i += 1
        if i == n:
            break
    return seqs

if __name__ == "__main__":
    n = 2
    T = 10.
    intensity_hawkes      = IntensityHawkes(mu=1, alpha=0.3, beta=1)
    intensity_poly        = IntensityPoly(segs=[0, T/4., T*2./4., T*3./4., T],
                                          b=0, A=[2, -2, 2, -2])
    intensity_hawkes_poly = IntensityHawkesPlusPoly(mu=1, alpha=0.3, beta=1,
                                                    segs=[0, T/4, T*2/4, T*3/4, T],
                                                    b=1, A=[1, -1, 1, -1])
    intensity_hawkes_gaussianmixture = IntensityHawkesPlusGaussianMixture(mu=1, alpha=0.3, beta=1,
                                                                          k=2, centers=[T/4, T*3/4], stds=[1, 1], coefs=[1, 1])

    print generate_sample(intensity_poly, T, n)
