# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:43:02 2021

@author: vcheplyg
"""

# Import packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, spatial
import pickle
import sys


path_repo = "/mnt/c/Users/doju/OneDrive - ITU/Research/cats-scans/"
sys.path.append(path_repo)

# Calculate transferability - move this to functions later
def get_transfer_score(auc_target, auc_source):
    transfer_score = (auc_source - auc_target) / auc_target * 100
    return transfer_score


# Create transferability plot like in the paper "Geometric Dataset Distances via Optimal Transport"
def plot_distance_score(distance, score_mean, score_std, labels, plot_name, dist_std=None, source=False):

    # Set color map for points, either targets or sources 
    df = labels.str.split(expand=True,)
    cm = {'imagenet': 'tab:purple', 'stl10': 'tab:brown', 'sti10': 'tab:pink', 'textures': 'tab:blue', 'isic': 'tab:red', 'chest': 'tab:orange', 'pcam-middle': 'tab:green', 'pcam-small': 'tab:gray', 'kimia': 'tab:olive'}
    if source:
        colors = df[0].map(cm)
    else:
        colors = df[2].map(cm)
    
    plt.figure(figsize=(8, 6))
    # plt.ylim([-10,20])
    plt.xlim([-0.05, 1.05])
   
    # Offset for labels
    offset_x = 0.01
    np.random.seed(1)
   
   # Plot the data

    for i, label in enumerate(labels):
        offset_y = np.random.rand(1)[0]
        plt.errorbar(
            distance[i,],
            score_mean[i,],
            yerr=score_std[i,],
            xerr=None if dist_std is None else dist_std[i],
            fmt="o",
            c=colors[i],
            ecolor=colors[i],
            label=label.split()[0] if source else label.split()[2]
        )
        plt.annotate(label, (distance[i] + offset_x, score_mean[i] + offset_y), size=8)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(distance, score_mean)
    str = "rho={:1.2f}, p={:1.2f}".format(r_value, p_value)
    sns.regplot(distance, score_mean, label=[str])
    plt.plot([], [], ' ', label=' ')

    # Plot details
    plt.xlabel("Dataset distance")
    plt.ylabel("Relative AUC increase")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles = list(by_label.values())[2:] + list(by_label.values())[:2]
    labels = list(by_label.keys())[2:] + list(by_label.keys())[:2]
    plt.legend(handles, labels, title = 'Source' if source else 'Target', loc='lower left')
    plt.tight_layout()
    plt.savefig(path_repo + "figures/" + plot_name + ".png")

def show_transfer_matrix(scores, labels):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(scores, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(x=j, y=i, s=scores[i, j], va="center", ha="center", size="xx-large")

    ax.set_xticklabels([""] + labels)
    ax.set_yticklabels([""] + labels)


def test_transfer_matrix():

    dataset_distance = [5, 10, 15, 20, 25, 30]

    score_mean = [7, 9, 13, 14, 11, 16]
    score_std = [1, 1, 1, 1, 1, 1]

    labels_pairs = ["A to B", "A to C", "A to D", "B to C", "B to D", "C to D"]
    plot_distance_score(dataset_distance, score_mean, score_std, labels_pairs)

    labels = ["A", "B", "C", "D"]
    scores = np.matrix(
        [[70, 75, 80, 75], [60, 70, 60, 80], [90, 95, 80, 85], [80, 60, 60, 65]]
    )

    show_transfer_matrix(scores, labels)


#################  Load transfer experiment AUCs
aucs = pd.read_csv(path_repo + "results/auc_folds_means_std.csv")
aucs = aucs.dropna(axis=0)


#  Add columns for transfer scores and distances
num_folds = 5
for fold in np.arange(0, num_folds) + 1:

    col = "score_" + str(fold)
    aucs[col] = np.nan

    col = "distance_" + str(fold)
    aucs[col] = np.nan


# Calculate transferability
for index, row in aucs.iterrows():

    target = row["target"]

    # Calculate transfer score

    for fold in np.arange(0, num_folds) + 1:

        baseline = aucs.loc[(aucs["target"] == target) & (aucs["source"] == target)]

        col = str(fold)

        target_only = baseline["fold_" + col]
        with_transfer = row["fold_" + col]

        score = get_transfer_score(target_only, with_transfer)

        aucs.at[index, "score_" + col] = score

aucs.to_csv(path_repo + "results/aucs_scores.csv")

########################### Transferability vs task2vec distance


# This is where the embedded subsets live
path_emb = path_repo + "results/Task2Vec_embeddings/"

# Let us generate some subsets to calculate the distances of
num_subsets = 100  # This is how many subsets there were already per dataset
num_folds = 100  # This is how many subsets to include in the figure

subset_index = np.random.randint(1, num_subsets, num_folds)

os.chdir(path_repo)  # magic required to load pickles

embeds = pd.read_csv(path_repo + "results/task2vec_embeddings.csv")

# Use the aucs table to decide what the source and target are
for index, row in aucs.iterrows():

    target = row["target"]
    source = row["source"]

    if (
        source == "imagenet"
    ):  # We do not have task2vec distances of ImageNet due to its size so we use STL10 which is a subset of ImageNet instead
        source = "stl10"

    for fold in np.arange(0, num_folds) + 1:

        # Load those subsets
        col = str(fold)
        subset = str(subset_index[fold - 1])

        # This is loading pickle files in results/Task2Vec_embeddings
        path_target = path_emb + "embedding_" + target + "_subset" + subset + ".p"
        path_source = path_emb + "embedding_" + source + "_subset" + subset + ".p"

        emb_target = pickle.load(open(path_target, "rb"))
        emb_source = pickle.load(open(path_source, "rb"))

        # Find distance
        dist = spatial.distance.cosine(emb_target.hessian, emb_source.hessian)
        aucs.at[index, "distance_" + col] = dist


aucs.to_csv(path_repo + "results/aucs_scores_distances.csv")

# Make plot?
distance_col = [col for col in aucs if col.startswith("distance")]
score_col = [col for col in aucs if col.startswith("score")]

labels = aucs["source"] + " to " + aucs["target"]
labels = labels.reset_index(drop=True)

"""
Distances normalized, but they are already between 0 and 1 ?
"""
# dist = aucs[distance_col]
# dist_norm = dist / np.max(dist)
dist = aucs[distance_col].mean(axis=1).to_numpy()
stddist = aucs[distance_col].std(axis=1).to_numpy()


meanauc = aucs[score_col].mean(axis=1).to_numpy()
stdauc = aucs[score_col].std(axis=1).to_numpy()


plot_distance_score(dist, meanauc, stdauc, labels, "transferability_task2vec", dist_std=stddist, source=False)


########################### Transferability vs expert distance

# Load expert distances
expdist = pd.read_csv(path_repo+'results/experts_clean.csv', sep=";")

# Initialize column in same dataframe
aucs['expert_distance'] = np.nan

# Lookup expert distances from CSV
for index, row in aucs.iterrows():
    
    target = row['target']
    source = row['source']
   
    
    dist = expdist.loc[(expdist['source']==source) & (expdist['target']==target)]
    aucs.at[index, 'expert_distance'] = dist['distance']

         
aucs.to_csv(path_repo+'results/aucs_scores_distances_experts.csv')


dist = aucs['expert_distance'].to_numpy()
dist = dist / np.max(dist)


plot_distance_score(dist, meanauc, stdauc, labels, 'trasferability_experts', source=False)

