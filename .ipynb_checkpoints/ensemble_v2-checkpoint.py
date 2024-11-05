from models.modules import ConformerModel, ChunkConformerOptModel
from models.utils import (
    calculate_ranking_metrics,
    dataloader_factory,
    dir_checker,
    reduce_func,
    reduce_func_500,
    save_best_log,
    save_logs,
    save_predictions,
    save_test_predictions,
    calculate_ndcg,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import yaml

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config["train"]["dataset_path"] = "/home/jovyan/artemvazhentsev/test_exp/data/train"
config["val"]["dataset_path"] = "/home/jovyan/artemvazhentsev/test_exp/data/train"
config["test"]["dataset_path"] = "/home/jovyan/artemvazhentsev/test_exp/data/test"

import torch

path = "/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-10-30/20-52-59/model"
# path = "/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-10-31/16-47-52-chunk_mixup_ls/model"
models = []
for filename in os.listdir(path):
    model = ChunkConformerOptModel(num_classes=config["model"]["num_classes"], 
                           emb_dim=config["model"]["emb_dim"],
                           num_layers=4)
    model.load_state_dict(torch.load(f"{path}/{filename}", weights_only=True, map_location="cuda"))
    models.append(model)

test_loader = dataloader_factory(config=config, data_split="test")

from tqdm import tqdm
from sklearn.metrics import pairwise_distances_chunked
import torchaudio
from transformers import set_seed

set_seed(42)

f_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
t_masking = torchaudio.transforms.TimeMasking(time_mask_param=10)

def reduce_func(D_chunk, start):
    top_size = 500
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]

def validation_step(model, batch):
    anchor_id = batch["anchor_id"]
    
    feats = []
    features = model.forward(batch["anchor"].to(config["device"]))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(torch.flip(batch["anchor"].to(config["device"]), dims=[2]))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(f_masking(t_masking(batch["anchor"].to(config["device"]))))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(f_masking(t_masking(batch["anchor"].to(config["device"]))))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(torch.flip(f_masking(t_masking(batch["anchor"].to(config["device"]))), dims=[2]))
    feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(torch.roll(batch["anchor"].to(config["device"]), roll, dims=[2]))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(torch.flip(torch.roll(batch["anchor"].to(config["device"]), roll, dims=[2]), dims=[2]))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(f_masking(t_masking(torch.roll(batch["anchor"].to(config["device"]), roll, dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(f_masking(t_masking(torch.flip(torch.roll(batch["anchor"].to(config["device"]), roll, dims=[2]), dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(f_masking(t_masking(torch.roll(torch.flip(batch["anchor"].to(config["device"]), dims=[2]), roll, dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())

    
    feats = np.concatenate(feats, axis=1)

    return {
        "anchor_id": anchor_id.numpy(),
        "f_c": torch.from_numpy(feats),
    }
        
def test(model, test_loader, split="test", return_preds = False):
    test_results = {}
    
    model.eval()
    trackids = []
    embeddings = []
    for batch in tqdm(test_loader):
        test_dict = validation_step(model, batch)
        if test_dict["f_c"].ndim == 1:
            test_dict["f_c"] = test_dict["f_c"].unsqueeze(0)
        for anchor_id, embedding in zip(test_dict["anchor_id"], test_dict["f_c"]):
            trackids.append(anchor_id)
            embeddings.append(embedding.numpy())
    predictions = []
    for chunk_result in tqdm(pairwise_distances_chunked(embeddings, metric='cosine', reduce_func=reduce_func, working_memory=1000)):
        for query_indx, query_nearest_items in chunk_result:
            predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
            
    if return_preds:
        return trackids, embeddings, predictions, test_loader

config["test"]["batch_size"] = 256

preds = {}
for i, model in tqdm(enumerate(models)):
    trackids, embeddings, predictions, test_loader = test(model.to(config["device"]), test_loader, return_preds=True)
    preds[i] = predictions

def average_predictions(preds):
    averaged_predictions = []
    for prediction in range(len(preds[0])):
        final_weight = {}
        for fold in range(len(preds)):
            weights = {k:1/np.sqrt(i+1) for i,k in enumerate(np.array(preds[fold][prediction][1]))}
            for key in list(weights.keys()):
                final_weight[key] = final_weight.get(key, 0) + weights.get(key, 0)

        new_order = [k for k, v in sorted(final_weight.items(), key=lambda item: -item[1])][:100]
        averaged_predictions.append((preds[fold][prediction][0], new_order))
    return averaged_predictions

import os

def save_test_predictions(predictions, output_dir: str, fn = None, n_items=100) -> None:
    if fn is None:
        filename = 'submission.txt'
    else:
        filename = f'{fn}.txt'
    with open(os.path.join(output_dir, filename), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest[:n_items]))))
            

averaged_predictions = average_predictions(preds)
# save_test_predictions(averaged_predictions, output_dir="./workdir/ensemble", fn="ensemble_1_augs_v1.1")
save_test_predictions(averaged_predictions, output_dir="./workdir/ensemble", fn="ensemble_1_augs_v1.2")