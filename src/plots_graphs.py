import matplotlib.pyplot as plt
import pandas as pd


def plot_epochs(metric, ylab):
    """
    Plot the metric for the CASIA2 dataset
    :param metric: The metric that we want to plot for the CASIA2 dataset
    :param ylab: The label of the y axis
    """
    plt.plot(metric, label='CASIA2')
    plt.ylabel(ylab)
    plt.xlabel("Epoch")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    df1 = pd.read_csv(filepath_or_buffer="SRM_accuracy.csv")
    df3 = pd.read_csv(filepath_or_buffer="SRM_loss.csv")
    plot_epochs(df1.iloc[:, 1], 'Training Accuracy')
    plot_epochs(df3.iloc[:, 1], 'Training Loss')
