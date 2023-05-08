#Moutsiounas Panagiotis
#The friedman test and the test nemenyi.
import pandas as pd
import scipy.stats as stats


data = pd.read_csv("algo_performance.csv", header=0, index_col=0)


algorithm_names = data.columns.tolist()


data = data.transpose()


data = data.reset_index()


data = data.rename(columns={'index': 'algorithm'})

# Set the names of the algorithms
data['algorithm'] = algorithm_names


grouped_data = data.groupby('algorithm')['C4.5']
grouped_data = data.groupby('algorithm')[algorithm_names[0]]



friedman_test = stats.friedmanchisquare(*[group[algorithm_names[0]] for name, group in grouped_data])



print("Friedman test statistic: {:.3f}".format(friedman_test.statistic))
print("Friedman test p-value: {:.3f}".format(friedman_test.pvalue))


nemenyi_test = stats.posthoc_nemenyi_friedman(data, val_col='0', group_col='algorithm')


print("\nNemenyi test result:")
print(nemenyi_test)