import glob
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
import seaborn as sb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import math
import json
import argparse




def get_ssim_score(image_A, image_B):
    (score, diff) = ssim(image_A, image_B, full=True, gaussian_weights=False,  use_sample_covariance=True)
    return score

def write_json(uid, input_channel_ids, target_channel_ids, path_to_save):
    json_data = {
        "uid": uid,
        "source_channel_ids": input_channel_ids,
        "target_channel_ids": target_channel_ids
    }
    with open(path_to_save + uid + '.json', 'w') as fp:
        json.dump(json_data, fp)

def save_plots(corr, Z, df):
    # Heatmap Plot
    sb.heatmap(corr, cmap="Blues", annot=True)
    plt.savefig('plots/correlation_based_on_SSIM.png')
    
    #Dedrogram Plot
    plt.figure(figsize=(14,6))
    plt. clf()
    dendrogram(Z, labels=df.columns, orientation='top', 
                leaf_rotation=90)
    plt.title('Dendrogram based on SSIM', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.gcf().set_size_inches(5.5, 2.5)
    plt.tight_layout()
    plt.savefig('plots/Dendrogram.pdf', dpi=300,  format='pdf')


def get_cluster(files, channel_names, plot = True):
    dfs = []
    for counter, filename in enumerate(files):
        print(counter)
        this_tiff = io.imread(filename)
        for idx, channel_name in enumerate(channel_names):
            for idx2 in range(this_tiff.shape[0]):
                score = get_ssim_score(this_tiff[idx,:,:], this_tiff[idx2,:,:])
                channel_ssims[channel_name].append(score)
        df = pd.DataFrame.from_dict(channel_ssims)
        dfs.append(df)
                
    df = pd.concat(dfs).groupby(level=0).mean()

    corr = df.corr()
    dissimilarity = 1 - abs(corr)
    Z = linkage(squareform(dissimilarity), 'complete')

    if plot:
        save_plots(corr, Z, df)
    return Z

def get_channel_wise_cluster(numclust, threshold = 0.70):
    Z = get_cluster(files, channel_names, plot=True)
    
    if numclust:
        labels = fcluster(Z,numclust,criterion='maxclust')
    elif threshold:
        labels = fcluster(Z, threshold, criterion='distance')  
    else:
        print('Neither threshold nor numcluster provided for clustering')
    return labels

def write_channel_selection_json(cluster_wise_channel_id, percentage_for_input = 0.7):
    input_channel_ids = []
    target_channel_ids = []
    for cluster_id in cluster_wise_channel_id:
        input_fraction = len(cluster_wise_channel_id[cluster_id]) * percentage_for_input
        input_fraction = math.floor(input_fraction)

        input_channel_ids.extend(cluster_wise_channel_id[cluster_id][:input_fraction])
        target_channel_ids.extend(cluster_wise_channel_id[cluster_id][input_fraction:])

    uid = f"cluster_{percentage_for_input:.2f}_{len(input_channel_ids)}"


    write_json(uid, input_channel_ids, target_channel_ids, path_to_save = 'channel_id_distribution/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--channel_names', type=str, required=True)
    parser.add_argument('--num_of_cluster', type=int, required=True)
    parser.add_argument('--percentage_from_each_cluster', type=float, required=True)

    args = parser.parse_args()
    


    data_dir = args.data_dir
    channel_names_dir = args.channel_names
    numclust = args.num_of_cluster
    percentage_from_each_cluster =  args.percentage_from_each_cluster

    files = glob.glob(data_dir + '*.tif')
    filse = files[:1]

    with open(channel_names_dir) as fp:
        lines = fp.readlines()
    channel_names = [line.strip() for line in lines]
    print(channel_names)
    channel_ssims = {channel_name: [] for channel_name in channel_names}

    
    labels = get_channel_wise_cluster(numclust, threshold = 0.70) 

    cluster_wise_channel_id = {cluster_id: [] for cluster_id in range(1,numclust+1)}
    print(labels)

    for index, channel_id in enumerate(labels):
        cluster_wise_channel_id[channel_id].append(index)


    percentage_from_each_cluster = 0.7
    write_channel_selection_json(cluster_wise_channel_id, percentage_from_each_cluster)

    #python select_channel_from_cluster.py --data_dir ~/Dataset/codex_data/raw_data_scaled/ --channel_names channel_names.txt --num_of_cluster 6 --percentage_from_each_cluster 0.8





