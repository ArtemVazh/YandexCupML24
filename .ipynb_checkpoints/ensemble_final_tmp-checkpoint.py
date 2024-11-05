from models.modules import ConformerModel, ChunkConformerOptModel, ChunkConformerOptV5Model
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

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed

import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_chunked

set_seed(42)
f_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
t_masking = torchaudio.transforms.TimeMasking(time_mask_param=10)

f_masking1 = torchaudio.transforms.FrequencyMasking(freq_mask_param=63)
t_masking1 = torchaudio.transforms.TimeMasking(time_mask_param=33)

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

    for roll in [10, 20, 40]:
        features = model.forward(f_masking1(t_masking1(torch.roll(batch["anchor"].to(config["device"]), roll, dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(f_masking1(t_masking1(torch.flip(torch.roll(batch["anchor"].to(config["device"]), roll, dims=[2]), dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(f_masking1(t_masking1(torch.roll(torch.flip(batch["anchor"].to(config["device"]), dims=[2]), roll, dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())

    for roll in [10, 20, 40]:
        features = model.forward(f_masking1(t_masking1(torch.roll(torch.flip(batch["anchor"].to(config["device"]), dims=[2]), roll, dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())
        
    for roll in [10, 20, 40]:
        features = model.forward(f_masking1(t_masking1(torch.roll(torch.flip(batch["anchor"].to(config["device"]), dims=[2]), roll, dims=[2]))))
        feats.append(features["f_c"].detach().cpu().numpy())
    
    features = model.forward(torch.flip(f_masking1(t_masking1(batch["anchor"].to(config["device"]))), dims=[2]))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(torch.flip(f_masking1(t_masking1(batch["anchor"].to(config["device"]))), dims=[2]))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(torch.flip(f_masking1(t_masking1(batch["anchor"].to(config["device"]))), dims=[2]))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(f_masking1(t_masking1(batch["anchor"].to(config["device"]))))
    feats.append(features["f_c"].detach().cpu().numpy())

    features = model.forward(f_masking1(t_masking1(batch["anchor"].to(config["device"]))))
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
    for chunk_result in tqdm(pairwise_distances_chunked(embeddings, metric='cosine', reduce_func=reduce_func, working_memory=1000, n_jobs=16)):
        for query_indx, query_nearest_items in chunk_result:
            predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
            
    if return_preds:
        return trackids, embeddings, predictions, test_loader

def average_predictions(preds, top_items=100):
    averaged_predictions = []
    for prediction in range(len(preds[0])):
        final_weight = {}
        for fold in range(len(preds)):
            weights = {k:1/np.sqrt(i+1) for i,k in enumerate(np.array(preds[fold][prediction][1]))}
            for key in list(weights.keys()):
                final_weight[key] = final_weight.get(key, 0) + weights.get(key, 0)

        new_order = [k for k, v in sorted(final_weight.items(), key=lambda item: -item[1])][:top_items]
        averaged_predictions.append((preds[fold][prediction][0], new_order))
    return averaged_predictions


def save_test_predictions(predictions, output_dir: str, fn = None, n_items=100) -> None:
    if fn is None:
        filename = 'submission.txt'
    else:
        filename = f'{fn}.txt'
    with open(os.path.join(output_dir, filename), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest[:n_items]))))


with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config["train"]["dataset_path"] = "/home/jovyan/artemvazhentsev/test_exp/data/train"
config["val"]["dataset_path"] = "/home/jovyan/artemvazhentsev/test_exp/data/train"
config["test"]["dataset_path"] = "/home/jovyan/artemvazhentsev/test_exp/data/test"
config["test"]["batch_size"] = 256
# config["test"]["max_size"] = -1

ensemble_filename = "ensemble_final_v1_tmp"

test_loader = dataloader_factory(config=config, data_split="test")

paths = ["/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-10-30/20-52-59/model",
         "/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-10-31/16-47-52-chunk_mixup_ls/model",
         "/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-11-02/01-16-12-chunk_mixup_ls_overall/model",
         "/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-11-02/13-59-37-chunk_mixup_ls_overall_seed777/model",
         "/home/jovyan/artemvazhentsev/test_exp/code/workdir/2024-11-02/13-59-36-chunk_mixup_ls_overall_seed123456789/model"]

model_types = ["chunk_opt", "chunk_opt", "chunk_opt_v5", "chunk_opt_v5", "chunk_opt_v5"]
cache_path = "./workdir/cache_predictions/"

ensemble_preds = {}
for k, (path, model_type) in enumerate(zip(paths, model_types)):
    models = []
    for filename in os.listdir(path):
        # if ("fold=6" in filename) and ("seed777" in path):
        #     print(f"skip {path}/{filename}")
        #     continue
        if ("fold=5" in filename) and ("seed123456789" in path):
            print(f"skip {path}/{filename}")
            continue
        if ("fold=6" in filename) and ("seed123456789" in path):
            print(f"skip {path}/{filename}")
            continue
        # if ("fold=8" in filename) and ("2024-11-02/01-16-12-chunk_mixup_ls_overall" in path):
        #     print(f"skip {path}/{filename}")
        #     continue
        
        if model_type == "chunk_opt":
            model = ChunkConformerOptModel(num_classes=config["model"]["num_classes"], 
                                   emb_dim=config["model"]["emb_dim"],
                                   num_layers=4)
        elif model_type == "chunk_opt_v5":
            model = ChunkConformerOptV5Model(num_classes=config["model"]["num_classes"],
                                             emb_dim=config["model"]["emb_dim"],
                                             num_layers=4,
                                             crop_size=config["model"].get("crop_size", 10),
                                             step_size=config["model"].get("step_size", 5),
                                             pos=config["model"].get("pos", False),
                                             scale_attn=config["model"].get("scale_attn", False),
                                             stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
        try:
            model.load_state_dict(torch.load(f"{path}/{filename}", weights_only=True, map_location="cuda"))
        except Exception as e:
            print(f"Error {e} with model {path}/{filename}")
            model.load_state_dict(torch.load(f"{path}/{filename}", weights_only=True, map_location="cuda"), strict=False)
        models.append(model)
        
    predictions_folds = {}
    for i, model in tqdm(enumerate(models)):
        prediction_path = f"{cache_path}/prediction_model={k}_fold={i}.txt"
        if os.path.exists(prediction_path):
            print(f"path exists {prediction_path}")
            df = pd.read_csv(prediction_path, sep='\t', converters={"versions": eval}, header=None)
            predictions = [(row[1].values[0], row[1].values[1:]) for row in df.iterrows()]
        else:
            print(f"make predictions model={k} fold={i}")
            trackids, embeddings, predictions, test_loader = test(model.to(config["device"]), test_loader, return_preds=True)
            save_test_predictions(predictions, output_dir=cache_path, fn=f"prediction_model={k}_fold={i}", n_items=500)
        predictions_folds[i] = predictions

    print(f"averaging models {k}")
    ensemble_preds[k] = average_predictions(predictions_folds, top_items=500)
    # save_test_predictions(ensemble_preds[k], output_dir="./workdir/ensemble", fn=ensemble_filename+f"_part_{k}", n_items=100)
    # if k == 1:
    #     print(f"averaging final models <{k}")
    #     final_predictions = average_predictions(ensemble_preds, top_items=100)
    #     save_test_predictions(final_predictions, output_dir="./workdir/ensemble", fn=ensemble_filename+f"_part_0_1")
        
print(f"averaging final models")
final_predictions = average_predictions(ensemble_preds, top_items=100)
save_test_predictions(final_predictions, output_dir="./workdir/ensemble", fn=ensemble_filename)