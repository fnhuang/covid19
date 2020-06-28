from data_analysis import get_folder_spec, SeedTweetsAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import math

class Histogram():
    def __init__(self, data):
        self.data = self.transform_data(data)


    def transform_data(self, data):
        if isinstance(data, (list)):
            # do nothing
            return data

    def plot_histo_log(self, num_bins):
        self.data = [math.log10(fol) for fol in self.data]
        weights = np.ones_like(self.data) / len(self.data)
        # adding weights gives probability
        plt.hist(self.data, bins=num_bins, weights=weights)
        plt.xlabel("num of followers ($10^x$)")
        plt.show()

    def plot_histogram(self, num_bins):
        weights = np.ones_like(self.data) / len(self.data)
        # adding weights gives probability
        plt.hist(self.data, bins=num_bins, weights=weights)
        plt.show()

    def draw_cumulative_distribution(self, num_bins):
        # normed True ensures integral equal to 1, normed False calculates real counts
        # counts are probability if normed = True, real counts if normed = False
        # in the following case, normed True, or False does not change the final results
        counts, bin_edges = np.histogram(self.data, bins=num_bins, normed=True)
        cdf = np.cumsum(counts)
        # cdf[-1] is the sum of all counts
        prob = cdf/cdf[-1]
        print(bin_edges[0:5])
        print(prob[0:5])
        # bin_edges [0] is 0
        plt.plot(bin_edges[1:], prob)
        plt.show()


folder_specs = get_folder_spec()
sta = SeedTweetsAnalyzer(folder_specs["seed tweets"])

#requirements = {"location": "singapore", "followers": [50,2000]}
requirements = {"location": "singapore","followers": [0,float("inf")],"verified": False}
#requirements = {}
seed_users = sta.get_seed_users()
filtered_users = sta.filter_seed_users(requirements)
print(len(seed_users), len(filtered_users))
followers = [u.followers_count for u in filtered_users]
followers.sort(reverse=True)
hist = Histogram(followers)
hist.draw_cumulative_distribution(100)
#hist.plot_histogram(100)
hist.plot_histo_log(100)

# verified user: verified true
# social capitalists: friends 500, follower 500