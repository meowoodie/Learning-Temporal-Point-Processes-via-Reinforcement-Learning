from utils.plots import *
from utils.ppgen import *

T=15
n=100

# Choose your intensity here.
# intensity_hawkes      = IntensityHawkes(mu=1, alpha=0.3, beta=1)
# intensity_poly        = IntensityPoly(segs=[0, T/4, T*2/4, T*3/4, T],
#                                       b=0, A=[1, -1, 1, -1])
# intensity_hawkes_poly = IntensityHawkesPlusPoly(mu=2, alpha=0.5, beta=3,
#                                                 segs=[0, T/4, T*2/4, T*3/4, T],
#                                                 b=5, A=[2, -1, 2, -1])
intensity = IntensityHawkesPlusGaussianMixture(mu=1, alpha=0.3, beta=1,
                                               k=2, centers=[T/4, T*3/4], stds=[1, 1], coefs=[1, 1])

expert_seqs = generate_sample(intensity, T=T, n=n)
# Generate your sequences here
# Your code for leaner_seqs

expert_intdiff = get_integral_diffs(expert_seqs, intensity)
# Get integral diffs for leaner_seqs
# leaner_intdiff = integral_diffs(learner_seqs, intensity)

qqplot4intdiff(expert_intdiff)
# Plot qqplot for leaner_intdiff
# qqplot4intdiff(leaner_intdiff)
