import warnings
warnings.filterwarnings('ignore')

from pase.models.frontend import wf_builder
import librosa
import os
import torch
import json
import numpy as np
import argparse

global pase, labels
pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('FE_e199.ckpt', load_last=True, verbose=True)

labels_array = ["no", "yes"]


def path_helper(path_string: str) -> str:
    return os.path.join(os.getcwd(), path_string)

"""
    Returns the output of the pase model on a given trajectory
"""
def run_pase_on_file(user_name: str, time_interval: float):
    WAV_PATH = path_helper(f'{user_name}.wav')
    ANN_PATH = path_helper(f'{user_name}_annotations.json')

    # Compute the length of the entire audio segment both as a tensor and in
    # the number of seconds
    y, sr = librosa.load(WAV_PATH, sr=48000)
    SECONDS = librosa.get_duration(y=y, sr=sr)
    y = torch.tensor(y).view((1, 1, -1))

    print(f"Shape of entire audio before pase: {y.shape}")
    # total size will be (1, 256, 9409), which are 20484 frames of 256 dims each
    total = pase(y, device="cpu")
    print(f"Shape of entire audio after pase:  {total.shape}")

    NUM_FRAMES = y.shape[2]

    # Load in the annotations file
    annotations = json.load(open(ANN_PATH, "r"))
    start_anns = [None] * len(annotations.keys())

    data = []
    labels = []

    # Loop thorugh each yes, no, etc in the json annotations
    for key in annotations.keys():
        start = annotations[key][0]
        end   = annotations[key][1]
        word  = annotations[key][2]

        # See if a word is 'yes' or 'no'
        for i in range(0, len(labels_array)):
            if word.lstrip().rstrip() == labels_array[i]:
                j = start
                while j + time_interval <= (end + 0.01):
            
                    # Compute intervals of yes/no at the time size passed under
                    # arguments. Proportional calculations are below
                    s_proportion = j / SECONDS
                    e_proportion = (j + time_interval) / SECONDS
                    s_index = int(NUM_FRAMES * s_proportion)
                    e_index = int(NUM_FRAMES * e_proportion)
                    subset = y[:,:,s_index:e_index]

                    # Conduct a forward pass through the existing PASE network
                    result = pase(subset)

                    # Append the new sub annotation to the overall dataset as
                    # well as the label
                    data.append(result)
                    labels.append(i)

                    j += time_interval
                break

    # Create the data such that it's a numpy array for easier processing
    # of PCA later
    data = [d.detach().numpy() for d in data]
    data_len = len(data)
    data = np.array(data)
    data = data.reshape((data_len, -1))
    labels  = [np.array(l) for l in labels]

    assert data_len == len(labels)
    print(f"Total of {len(data)} items in the dataset")
    return (data, labels)

def pca_and_plot(dataset: list, labels: list, dim: int,
                 time_interval: float, key: str) -> None:

    assert dim == 2 or dim == 3

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import plotly
    import plotly.graph_objs as go

    if dim == 3:
        pca_dim = PCA(random_state=0).fit_transform(dataset)[:,:3]
    else:
        pca_dim = PCA(random_state=0).fit_transform(dataset)[:,:2]

    print(f"PCA Shape: {pca_dim.shape}")

    plotting_data = []
    no  = []
    yes = []
    for i in range(4):
        no.append([])
        yes.append([])

    # Add x, y, z, and index coordinates to a 2 dimensional array to properly
    # create all of the scatter points
    for i in range(0, pca_dim.shape[0]):
        label_idx = int(labels[i])
        if label_idx == 0:
            no[0].append(pca_dim[i,0])
            no[1].append(pca_dim[i,1])
            if dim == 3:
                no[2].append(pca_dim[i,2])
            no[3].append(i)
        if label_idx == 1:
            yes[0].append(pca_dim[i,0])
            yes[1].append(pca_dim[i,1])
            if dim == 3:
                yes[2].append(pca_dim[i,2])
            yes[3].append(i)

    # Create either 3D scatter points or 2D scatter points depending on the
    # argument input
    if dim == 3:
        trace_input = go.Scatter3d(
            x=no[0],
            y=no[1],
            z=no[2],
            text=no[3],
            name='no',
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={'size': 10, 'opacity': 1, 'color': 'blue'}
        )
        plotting_data.append(trace_input)

        trace_input = go.Scatter3d(
            x=yes[0],
            y=yes[1],
            z=yes[2],
            text=yes[3],
            name='yes',
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={'size': 10, 'opacity': 1, 'color': 'green',}
        )
        plotting_data.append(trace_input)
    else:
        trace_input = go.Scatter(
            x=no[0],
            y=no[1],
            text=no[3],
            name='no',
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={'size': 10, 'opacity': 1, 'color': 'blue'}
        )
        plotting_data.append(trace_input)

        trace_input = go.Scatter(
            x=yes[0],
            y=yes[1],
            text=yes[3],
            name='yes',
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={'size': 10, 'opacity': 1, 'color': 'green',}
        )
        plotting_data.append(trace_input)

    # Establish the layout of the image
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(x=1, y=0.5, font=dict(family="Courier New", size=25, color="black")),
        font = dict(family = " Courier New ", size = 15),
        autosize = False,
        width = 1000,
        height = 1000
    )

    # Create and save the figure
    plot_figure = go.Figure(data=plotting_data, layout=layout)
    plot_figure.write_image(f'{dim}dscatter_{time_interval}_{key}.png')

    return

if __name__ == "__main__":

    """
    python pase_visualization.py --key spaceinvaders_X549THSLUZ --dim 3 --ti 0.5
    """

    parser = argparse.ArgumentParser(description='Turn atari frames to have heat map gaze information')
    parser.add_argument('--key', help="Enter the MTurk key user provided. Ex: mspacman_JE5W3X5P3T")
    parser.add_argument('--dim', help="List the number of dimensions for PCA", type=int, default=2)
    parser.add_argument('--ti',  help="Time interval for the audio segments", type=float, default=0.2)
    args = parser.parse_args()

    user_name = f"{args.key}_4"

    (dataset, labels) = run_pase_on_file(user_name, args.ti)

    pca_and_plot(dataset, labels, args.dim, args.ti, user_name)
