import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# Generate Marked Spatio-Temporal point processes using Ogata's paper

# For discrete marks
# X = MSTPPSamples(num_seq, T_max, mark_vol):

# ====================================================================

class MSTPPGenerator:

    def __init__(self, num_seq, T, mark_vol, alpha, beta, mu, frequence, magnitude, shift, num_component, xlim, ylim, grid_time, grid_space):

        self.num_seq = num_seq
        self.T = T
        self.mark_vol = mark_vol
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.frequence = frequence
        self.magnitude = magnitude
        self.shift = shift
        self.num_component = num_component
        self.xlim = xlim
        self.ylim = ylim
        self.grid_time = grid_time
        self.grid_space = grid_space


    def MSTPPSamples(self):

        # As a simple example, the time and the locations are generated separately
        # First generate time, then given total number of events, generate the associated
        # locations and marks

        # Generate time
        sequence_time = []
        for i in range(self.num_seq):
            sequence_cur = self.accept_reject_sample()  # generate one sequence of time
            sequence_time.append(sequence_cur)

        # sequence_time is list

        # Padding
        event_len = [len(seq) for seq in sequence_time]
        seq_len = max(event_len)
        # print(seq_len)
        sequence_time_pad = np.zeros((self.num_seq, seq_len))
        for i in range(self.num_seq):
            sequence_time_pad[i, 0:event_len[i]] = np.array(sequence_time[i])  # (num_seq, seq_len)
        sequence_time_pad = np.expand_dims(sequence_time_pad, axis=2)

        # Generate locations
        # First generate density function
        prob_component, center, bandwidth = self.Mix_Gaussian_density()
        sequence_location = self.Samplelocation_mxiture_Gaussian(seq_len, prob_component, center, bandwidth)  # (num_seq, seq_len, 2)

        # Generate marks
        mark_embedding = self.SampleMark(seq_len)  # (num_seq, seq_len, mark_vol); one hot embedding for marks

        sequence = np.concatenate((sequence_time_pad, sequence_location, mark_embedding), axis=2)
        return sequence, sequence_time_pad, sequence_location, mark_embedding


# ====================================================================
    # Sample time

    def Haw_intensity(self, history, t):
        return self.alpha * self.beta * np.sum(np.exp(-self.beta * (t - np.array(history))))

    def constant_intensity(self, t):
        return self.mu

    def periodic_intensity(self, t):
        return self.magnitude * np.sin(self.frequence * t + self.shift)




    def accept_reject_sample(self):
        # to make it easy, focus on only one sequence
        t = 0
        sequence_time = []

        while t < self.T:
            # update maximum intensity, suppose a jump just occurs

            intensity_up = self.Haw_intensity(sequence_time, t) + self.mu + self.magnitude
            # new event
            t = t + np.random.exponential(scale=1 / intensity_up)

            # accept or reject
            unif_rv = np.random.uniform(0, 1, 1)
            intensity_cur = self.Haw_intensity(sequence_time, t) + self.constant_intensity(t) + self.periodic_intensity(t)
            if unif_rv <= intensity_cur/intensity_up:
                # accept
                sequence_time.append(t)

        return sequence_time


# ====================================================================
    # Sample locations

    def Samplelocation_mxiture_Gaussian(self, seq_len, prob_component, center, bandwidth):
        # centers: (num_component, 2)
        # bandwidth: (num_component, )
        # under the restrict independence assumption
        sequence_location = []
        num_sample = seq_len * self.num_seq
        label = np.random.multinomial(num_sample, prob_component, size=1)
        label = np.squeeze(label)
        for i in range(len(prob_component)):
            sample_cur = np.random.multivariate_normal(mean=center[:, i], cov=[[bandwidth[i], 0], [0, bandwidth[i]]],
                                                       size=label[i])  # array (n, 2)
            sequence_location.append(sample_cur)
        sequence_location_stack = np.vstack(sequence_location)
        np.random.shuffle(sequence_location_stack)  # check whether this function shuffles the data by rows
        location = np.reshape(sequence_location_stack, (self.num_seq, seq_len, -1))
        return location



    def Mix_Gaussian_density(self):
        # xlim: [x1, x2] indicates the boundary of x
        # ylim: [y1, y2] indicates the boundary of y
        prob_component = np.random.uniform(0, 1, self.num_component)
        prob_component = prob_component / np.sum(prob_component)
        center = []
        x_center = np.random.uniform(self.xlim[0], self.xlim[1], self.num_component)
        center.append(x_center)
        y_center = np.random.uniform(self.ylim[0], self.ylim[1], self.num_component)
        center.append(y_center)
        center = np.array(center)
        bandwidth = np.random.normal(0, 1, self.num_component)**2
        return prob_component, center, bandwidth

# ====================================================================
    # Assume that the marks are generated from the Markov Chain

    def SampleMark(self, seq_len):

        P = np.random.normal(loc=0, scale=1, size=(self.mark_vol, self.mark_vol))**2 # make it a bit heavy-tailed
        P = np.divide(P, np.reshape(np.sum(P, axis=1), [self.mark_vol, 1]))
        P_cum = np.cumsum(P, axis=1)


        # Generate initial distribution
        I = np.random.rand(self.mark_vol)
        I = np.divide(I, np.sum(I))
        I_cum = np.cumsum(I)


        # Generate Markov chain (output: one-hot embedding)

        mark_embedding = np.zeros((self.num_seq, seq_len, self.mark_vol))
        markov_chain_prob = np.random.uniform(0, 1, (self.num_seq, seq_len, 1))
        Index = []
        for i in range(seq_len):
                if i == 0:
                    Index = np.tile(markov_chain_prob[:,i, :], (1, self.mark_vol)) > I_cum
                    Index = np.sum(Index, axis=1)
                    for j in range(self.num_seq):
                        mark_embedding[:, i, :][j, Index[j]] = 1
                else:
                    P_cum_cur = P_cum[Index, :]
                    Index = np.tile(markov_chain_prob[:, i, :], (1, self.mark_vol)) > P_cum_cur
                    Index = np.sum(Index, axis=1)
                    for j in range(self.num_seq):
                        mark_embedding[:, i, :][j, Index[j]] = 1
        return mark_embedding


# ====================================================================
# Visualization tools


    def visualize_intensity(self, sequence_time):

        # sequence_time: (num_seq, seq_len, 1)
        sequence_time = np.squeeze(sequence_time)

        # save data
        np.savetxt('sequence_time_pad', sequence_time, delimiter=',')


        _, seq_len = np.shape(sequence_time)
        ts = np.arange(0, self.T, self.grid_time)  # ts are anchor points
        num_grid = len(ts)
        cum_count = np.zeros((self.num_seq, num_grid))

        for index, time_gird in enumerate(ts):
            cum_count[:, index] = np.sum(1 * (sequence_time < time_gird* np.ones(shape=(self.num_seq, seq_len))) * 1 * (sequence_time > np.zeros(shape=(self.num_seq, seq_len))), axis=1)
        counts = np.append(cum_count[:, 0:1], np.diff(cum_count), axis=1)
        # print(counts)
        empirical_intensity = np.mean(counts, axis=0)
        std_empirical_intensity = np.std(counts, axis=0)
        upper_empirical_intensity = empirical_intensity + std_empirical_intensity
        lower_empirical_intensity = empirical_intensity - std_empirical_intensity

        plt.plot(ts, empirical_intensity)
        plt.fill_between(ts, upper_empirical_intensity, lower_empirical_intensity, color='grey', alpha='0.5')

        plt.show()


    def compute_mask(self, sequence_time):

        # sequence_time: (num_seq, seq_len, 1)
        _, seq_len, _ = np.shape(sequence_time)
        mask = 1 * (sequence_time <= self.T * np.ones(shape=(self.num_seq, seq_len, 1))) *\
               1 * (sequence_time > np.zeros(shape=(self.num_seq, seq_len, 1)))

        return mask

    def visualize_location(self, sequence_location, mask):

        # For locations and to make it simple, we directly plot the exact locations of the generated points
        # sequence_location: (num_seq, seq_len, 2)
        # mask: (num_seq, seq_len, 1)

        mask_location = np.concatenate(np.concatenate((mask, mask), axis=2), axis=0)
        print(np.shape(mask_location))
        sequence_location = np.concatenate(sequence_location, axis=0)

        np.savetxt('sequence_location', sequence_location, delimiter=',')

        sequence_location_mask = np.multiply(sequence_location, mask_location)
        x = sequence_location_mask[:, 0]
        y = sequence_location_mask[:, 1]

        print(x[0:10])
        print(y[0:10])
        plt.scatter(x[x != 0], y[y != 0])
        plt.show()


    def visualize_mark(self, mark_embedding, mask):
        # For marks, we visualize the empirical distributions of marks
        # mark_embedding: (num_seq, seq_len, mark_vol)
        # mask: (num_seq, seq_len, 1)

        sequence_mark = np.concatenate(mark_embedding, axis=0)

        np.savetxt('sequence_mark', sequence_mark, delimiter=',')

        _, index = np.nonzero(sequence_mark)
        index_mask, _ = np.nonzero(np.concatenate(mask, axis=0))
        sequence_mark_mask = sequence_mark[index_mask, :]
        _, label = np.nonzero(sequence_mark_mask)
        plt.hist(label, bins=np.arange(self.mark_vol+1))
        plt.show()

# ====================================================================
# main
if __name__ == '__main__':

    # set parameters

    num_seq = 10
    T = 20
    mark_vol = 5
    alpha = 0.6
    beta = 1
    mu = 2
    frequence = 1
    magnitude = 1
    shift = 0.5
    num_component = 3
    xlim = [-5, 5]
    ylim = [-5, 5]
    grid_time = 0.1
    grid_space = 1

    generator = MSTPPGenerator(num_seq=num_seq, T=T, mark_vol=mark_vol,
                            alpha=alpha, beta=beta, mu=mu, frequence=frequence,
                            magnitude=magnitude, shift=shift, num_component=num_component,
                            xlim=xlim, ylim=ylim,
                            grid_time=grid_time, grid_space=grid_space)

    sequence, sequence_time_pad, sequence_location, mark_embedding = generator.MSTPPSamples()
    mask = generator.compute_mask(sequence_time_pad)
    generator.visualize_intensity(sequence_time_pad)
    generator.visualize_location(sequence_location, mask)
    generator.visualize_mark(mark_embedding, mask)

    # print(sequence_location)
    # print(mark_embedding)
    # print(sequence)
