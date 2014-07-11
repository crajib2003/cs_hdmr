import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import pylab

def mean(dataset):
    return sum(dataset) / len(dataset)

def r_squared(actual, ideal):
    actual_mean = np.mean(actual)
    ideal_dev = np.sum([(val - actual_mean)**2 for val in ideal])
    actual_dev = np.sum([(val - actual_mean)**2 for val in actual])

    return ideal_dev / actual_dev

def plot_log_means(datasets, labels, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()

    means = [np.log(mean(d)) for d in datasets]
    num_labels = [float(l) for l in labels]

    fit = pylab.polyfit(num_labels, means, 1)
    fit_fn = pylab.poly1d(fit)

    lin_approx = fit_fn(num_labels)
    r_sq = r_squared(means, lin_approx)
    
    secondary_num_labels = [num_labels[0] - num_labels[1]] + num_labels + [num_labels[-1] + num_labels[1]]

    ax.plot(
        num_labels, means, 'yo',
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
        t = plt.title(title)
        t.set_y(1.02)

def main():
    datasets = []
    labels = []    
    
    with open(sys.argv[1], 'rb') as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            labels.append(row[0])
            datasets.append([float(r) for r in row[1:]])

    plot_log_means(datasets, labels, xlabel="Sample Size", ylabel="log(Mean Relative Reconstruction Error in Cross-Validation)", title="log(Mean Reconstruction Error) vs. Sample Size (Correlated Polynomial)")
    out_file = sys.argv[2]
    plt.savefig(out_file)
    
if __name__ == '__main__':
    main()
