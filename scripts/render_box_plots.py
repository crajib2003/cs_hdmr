import sys
import csv
import matplotlib.pyplot as plt

def plot_boxplots(datasets, labels, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()

    ax.boxplot(datasets[1:]) # Try: patch_artist=True?
    plt.xticks(range(1, len(labels[1:]) + 1), labels[1:])
    
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

    out_file = sys.argv[2]
    
    with open(sys.argv[1], 'rb') as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            labels.append(row[0])
            datasets.append([float(r) for r in row[1:]])

    plot_boxplots(datasets, labels, xlabel="Sample Size", ylabel="Distribution of Relative Reconstruction Errors in Cross-Validation", title="Reconstruction Error Distribution vs. Sample Size (Correlated Polynomial)")
    plt.savefig(out_file)
    
if __name__ == '__main__':
    main()
