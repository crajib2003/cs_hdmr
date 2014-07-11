import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import pylab

def r_squared(actual, ideal):
    actual_mean = np.mean(actual)
    ideal_dev = np.sum([(val - actual_mean)**2 for val in ideal])
    actual_dev = np.sum([(val - actual_mean)**2 for val in actual])

    return ideal_dev / actual_dev

def plot_variances(datasets, labels, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()

    variances = [np.log(np.var(d)) for d in datasets]
    num_labels = [float(l) for l in labels]

    fit = pylab.polyfit(num_labels, variances, 1)
    fit_fn = pylab.poly1d(fit)

    lin_approx = fit_fn(num_labels)
    r_sq = r_squared(variances, lin_approx)
    
    secondary_num_labels = [num_labels[0] - num_labels[1]] + num_labels + [num_labels[-1] + num_labels[1]]

    ax.plot(
        num_labels, variances, 'yo',
        secondary_num_labels, fit_fn(secondary_num_labels), '--k'
    )
    
    ax.annotate(
        '$(r^2 = {0:.2f})$'.format(r_sq),
        (0.90, 0.495),
        xycoords='axes fraction'
    )

    plt.xlim([secondary_num_labels[0], secondary_num_labels[-1]])
    
    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=20)
    if ylabel is not None:
        plt.ylabel(ylabel, labelpad=20)
    if title is not None:
        plt.title(title)
        
    plt.show()

def main():
    datasets = []
    labels = []    
    
    with open(sys.argv[1], 'rb') as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            labels.append(row[0])
            datasets.append([float(r) for r in row[1:]])

    plot_variances(datasets, labels, xlabel="Sample Size", ylabel="log(Variance of Relative Error in Cross-Validation)", title="Sample Size vs. log(Variance of Error) (Deg. 6, Ord. 3 Polynomial Function)")

if __name__ == '__main__':
    main()
