import seaborn as sns
import matplotlib.pyplot as plt     

def plot_cm(cm, labels):
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax) #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('RL Attention')
    ax.set_ylabel('Human Attention')
    # ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels, fontsize=8, ha='center') #['business', 'health']
    ax.yaxis.set_ticklabels(labels, fontsize=8, va='center')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":	

    cm = [[1.7232,	6.9725,	4.6438,	3.1318,	14.6131, 6.341],
          [5.2602,	2.0894,	4.9439,	3.795,	11.1326, 5.5665],
          [4.4342,	6.4031,	1.864,	3.3403,	9.8647,	5.711],
          [4.5262,	6.4553,	5.4946,	1.7835,	13.3413, 5.9193],
          [4.2899,	10.7528, 5.1853, 3.5888, 3.2, 6.4884],
          [5.0713,	7.6992,	5.9692,	3.5095,	14.0975, 2.8]]

    labels = ['Asterix', 'Breakout', 'Centipede', 'MsPacman', 'Phoenix', 'Seaquest']
    plot_cm(cm, labels)
