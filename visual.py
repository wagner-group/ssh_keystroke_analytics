from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["figure.figsize"] = (14,8.5)


def plot_tfp(tfps, filename, param):

    plt.clf()
    X, Y = [], []
    for tfp in tfps:
        tfp.to('cpu')
        X.append([tfp[i,0].item() for i in range(tfp.shape[0])])
        Y.append([tfp[i,1].item() for i in range(tfp.shape[0])])

    # create the figure and axes objects
    fig, ax = plt.subplots()
    fig.set_dpi(333)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    def animate(i):
        ax.clear()
        ax.scatter(X[i], Y[i])
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        fig.suptitle(f"Model performance per user, for {param(i+1)}")

    ani = FuncAnimation(fig, animate, frames=len(tfps), interval=500, repeat=True)
    writergif = PillowWriter(fps=2)
    ani.save(filename, writer=writergif)


def plot_accuracy(Y, X, filename, param):

    plt.clf()
    plt.plot(X,Y)
    plt.xlabel(f"{param}")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy as a function of {param}")
    plt.savefig(filename, dpi=333)


def plot_ROC(fpr, tpr, thsd, filename, params, title="All users"):

    plt.clf()
    plt.figure()
    plt.plot(fpr,tpr,color="darkorange",lw=2,label=f"ROC (area = %0.2f)" % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {title} ({params})")
    plt.legend(loc="lower right")
    plt.savefig(filename, dpi=333)
    plt.close()
