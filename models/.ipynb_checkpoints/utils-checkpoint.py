import glob
import json
import os
import re
from typing import Dict, List, Tuple

import jsonlines
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from models.data_loader import cover_dataloader

from models.data_model import Postfix
from sklearn.metrics import ndcg_score

def reduce_func(D_chunk, start):
    top_size = 100
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]

def reduce_func_500(D_chunk, start):
    top_size = 500
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]

def dataloader_factory(config: Dict, data_split: str, versions=None, val_versions=None) -> List[DataLoader]:
    return cover_dataloader(
        data_path=config["data_path"],
        file_ext=config["file_extension"],
        data_split=data_split,
        max_len=50,
        versions=versions,
        val_versions=val_versions,
        **config[data_split]
    )
    
def calculate_ranking_metrics(embeddings: np.ndarray, cliques: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(embeddings, metric="cosine")
    s_distances = np.argsort(distances, axis=1)
    cliques = np.array(cliques)
    query_cliques = cliques[s_distances[:, 0]]
    search_cliques = cliques[s_distances[:, 1:]]

    query_cliques = np.tile(query_cliques, (search_cliques.shape[-1], 1)).T
    mask = np.equal(search_cliques, query_cliques)

    ranks = 1.0 / (mask.argmax(axis=1) + 1.0)

    cumsum = np.cumsum(mask, axis=1)
    mask2 = mask * cumsum
    mask2 = mask2 / np.arange(1, mask2.shape[-1] + 1)
    average_precisions = np.sum(mask2, axis=1) / np.sum(mask, axis=1)

    return (ranks, average_precisions)

def calculate_ndcg(predictions: np.ndarray, test_loader, k=100):
    dcg_all = []
    for preds in predictions:
        target = test_loader.dataset.versions.loc[test_loader.dataset.version2clique.loc[preds[0], "clique"], "versions"]
        preds_versions = preds[1]
        dcg = 0
        idcg = 0
        for i, p in enumerate(preds_versions[:k-1]):
            if p in target:
                dcg += 1 / np.sqrt(i+1)
            
        for i, p in enumerate(target[:k-1]):
            idcg += 1 / np.sqrt(i+1)
        if dcg:
            dcg_all.append(dcg / idcg)
        else:
            dcg_all.append(0)
    
    return np.mean(dcg_all)

def dir_checker(output_dir: str) -> str:
    return output_dir

def save_test_predictions(predictions: List, output_dir: str, fold = None, n_items=100) -> None:
    if fold is None:
        filename = 'submission.txt'
    else:
        filename = f'submission_fold_{fold}.txt'
    with open(os.path.join(output_dir, filename), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest[:n_items]))))

def save_predictions(outputs: Dict[str, np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key in outputs:
        if "_ids" in key:
            with jsonlines.open(os.path.join(output_dir, f"{key}.jsonl"), "w") as f:
                if len(outputs[key][0]) == 4:
                    for clique, anchor, pos, neg in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor, "positive_id": pos, "negative_id": neg})
                else:
                    for clique, anchor in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor})
        else:
            np.save(os.path.join(output_dir, f"{key}.npy"), outputs[key])


def save_logs(outputs: dict, output_dir: str, name: str = "log") -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}.jsonl")
    with jsonlines.open(log_file, "a") as f:
        f.write(outputs)


def save_best_log(outputs: Postfix, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "best-log.json")
    with open(log_file, "w") as f:
        json.dump(outputs, f, indent=2)
