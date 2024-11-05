import logging
import os
from copy import deepcopy
from typing import Dict, List
import itertools
import pandas as pd
import json

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import copy
import torchaudio

from models.data_model import BatchDict, Postfix, TestResults, ValDict
from models.early_stopper import EarlyStopper
from models.modules import ConformerModel, CNNModel, AST_Model, FocalLoss, MyNTXentLoss, ConformerModelMoE, TransformerModelMoE, EffNetModel, HardTripletLoss, ChunkConformerModel, ChunkConformerOptModel, ChunkTransformerCNNOptModel, ChunkConformerOptV2Model, ChunkConformerOptV3Model, ChunkConformerOptV4Model, TransformerFreqModel, ChunkConformerOptV5Model, ChunkConformerOptV6Model
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
    calculate_ndcg
)
from pytorch_metric_learning.losses import NTXentLoss
from sklearn.model_selection import KFold

import logging
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: Dict, workdir: str) -> None:
        self.config = config
        self.workdir = workdir
        self.run_dir = ""
        self.best_model_path: str = None
        self.folds_best_model_path = {}
        self.predictions = {}
        self.num_classes = self.config["model"]["num_classes"]
        self.max_len = 50
        self.focal_loss = FocalLoss(gamma=self.config["train"]["gamma"], label_smoothing=self.config["train"]["label_smoothing"])
        if self.config["train"].get("use_my_loss", False):
            self.ntxent_loss = MyNTXentLoss
        else:
            self.ntxent_loss = NTXentLoss()

        self.early_stop = EarlyStopper(patience=self.config["train"]["patience"])
        self.state = "running"

        if self.config["train"].get("conformer", True):
            self.model_init = lambda config: ConformerModel(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            cls=config["model"].get("cls", "arcface")).to(config["device"])
        else:
            if self.config["train"].get("model_name", None) == "ast":
                self.model_init = lambda config: AST_Model(num_classes=config["model"]["num_classes"]).to(config["device"])
            elif self.config["train"].get("model_name", None) == "moe":
                self.model_init = lambda config: ConformerModelMoE(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"]).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk":
                self.model_init = lambda config: ChunkConformerModel(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"]).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_opt":
                self.model_init = lambda config: ChunkConformerOptModel(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            crop_size=config["model"].get("crop_size", 10),
                                                            step_size=config["model"].get("step_size", 5),
                                                            pos=config["model"].get("pos", False),
                                                            scale_attn=config["model"].get("scale_attn", False),
                                                            stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_opt_v2":
                self.model_init = lambda config: ChunkConformerOptV2Model(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            crop_size=config["model"].get("crop_size", 10),
                                                            step_size=config["model"].get("step_size", 5),
                                                            pos=config["model"].get("pos", False),
                                                            scale_attn=config["model"].get("scale_attn", False),
                                                            stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_opt_v3":
                self.model_init = lambda config: ChunkConformerOptV3Model(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            crop_size=config["model"].get("crop_size", 10),
                                                            step_size=config["model"].get("step_size", 5),
                                                            pos=config["model"].get("pos", False),
                                                            scale_attn=config["model"].get("scale_attn", False),
                                                            stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_opt_v4":
                self.model_init = lambda config: ChunkConformerOptV4Model(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            crop_size=config["model"].get("crop_size", 10),
                                                            step_size=config["model"].get("step_size", 5),
                                                            pos=config["model"].get("pos", False),
                                                            scale_attn=config["model"].get("scale_attn", False),
                                                            stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_opt_v5":
                self.model_init = lambda config: ChunkConformerOptV5Model(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            crop_size=config["model"].get("crop_size", 10),
                                                            step_size=config["model"].get("step_size", 5),
                                                            pos=config["model"].get("pos", False),
                                                            scale_attn=config["model"].get("scale_attn", False),
                                                            stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_opt_v6":
                self.model_init = lambda config: ChunkConformerOptV6Model(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"],
                                                            crop_size=config["model"].get("crop_size", 10),
                                                            step_size=config["model"].get("step_size", 5),
                                                            pos=config["model"].get("pos", False),
                                                            scale_attn=config["model"].get("scale_attn", False),
                                                            stat_pooling=config["model"].get("stat_pooling", False)).to(config["device"])
            elif self.config["train"].get("model_name", None) == "chunk_cnn_trans":
                self.model_init = lambda config: ChunkTransformerCNNOptModel(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"]).to(config["device"])
                
            elif self.config["train"].get("model_name", None) == "trans_freq":
                self.model_init = lambda config: TransformerFreqModel(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"]).to(config["device"])
            elif self.config["train"].get("model_name", None) == "trans_moe":
                self.model_init = lambda config: TransformerModelMoE(num_classes=config["model"]["num_classes"], 
                                                            emb_dim=config["model"]["emb_dim"],
                                                            dropout=config["model"]["dropout"],
                                                            num_layers=config["model"]["num_layers"], 
                                                            num_heads=config["model"]["num_heads"]).to(config["device"])

            elif self.config["train"].get("model_name", None) == "effnet":
                self.model_init = lambda config: EffNetModel(num_classes=config["model"]["num_classes"]).to(config["device"])
            else:
                self.model_init = lambda config: CNNModel(num_classes=config["model"]["num_classes"]).to(config["device"])

        if self.config.get("new_loader", False):
            self.test_loader = dataloader_factory_v2(config=self.config, data_split="test")
        else:
            self.test_loader = dataloader_factory(config=self.config, data_split="test")
        self.postfix = {}
        self.folds_fina_data = {}

    def pipeline(self) -> None:
        self.run_dir = dir_checker(self.workdir)
        self.versions = pd.read_csv(os.path.join(self.config['data_path'], "cliques2versions.tsv"), sep='\t', converters={"versions": eval})
        if self.config["train"]["max_size"] != -1:
            self.versions = self.versions.sample(self.config["train"]["max_size"])

        cliques = self.versions["clique"].values
        kfolds = KFold(n_splits=self.config["train"]["n_folds"], shuffle=True, random_state=self.config["seed"])
        
        for fold, (train_index, val_index) in enumerate(kfolds.split(cliques)):
            self.postfix["Fold"] = fold

            self.model = self.model_init(self.config)
            optimizer = self.configure_optimizers(self.model)
            if self.config["device"] != "cpu":
                self.scaler = torch.amp.GradScaler('cuda', enabled=self.config["train"]["mixed_precision"])
            
            train_cliques = self.versions.iloc[train_index]
            val_cliques = self.versions.iloc[val_index]

            if self.config.get("new_loader", False):
                train_loader = copy.deepcopy(dataloader_factory_v2(config=self.config, data_split="train", versions=train_cliques))
                val_loader = copy.deepcopy(dataloader_factory_v2(config=self.config, data_split="val", versions=val_cliques))
            else:   
                train_loader = copy.deepcopy(dataloader_factory(config=self.config, data_split="train", versions=train_cliques, val_versions=copy.deepcopy(val_cliques)))
                val_loader = copy.deepcopy(dataloader_factory(config=self.config, data_split="val", versions=val_cliques))
            
            for epoch in tqdm(range(self.config["train"]["epochs"])):
                self.postfix["Epoch"] = epoch
                self.train_procedure(self.model, optimizer, train_loader, val_loader, epoch)
                if self.state == "early_stopping":
                    break

            self.folds_fina_data[fold] = self.postfix.copy()
            self.early_stop = EarlyStopper(patience=self.config["train"]["patience"])
            self.test_model(fold)
            self.best_model_path = None
            self.model = None
            self.state = "running"
            
        self.average_predictions()
        with open(f'{self.run_dir}/meta_data.json', 'w') as fp:
            json.dump(self.folds_fina_data, fp)
    
    def train_procedure(self, model, optimizer, train_loader, val_loader, epoch) -> None:
        model.train()
        
        train_loss_list = []
        train_cls_loss_list = []
        train_sim_loss_list = []
        train_sim_test_loss = []
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), position=2, leave=False):
            train_step = self.training_step(model, optimizer, batch, train_loader, epoch, step)
            train_loss_list.append(train_step["train_loss_step"])
            train_cls_loss_list.append(train_step["train_cls_loss"])
            train_sim_loss_list.append(train_step["train_sim_loss"])
            train_sim_test_loss.append(train_step["train_sim_test_loss"])

            if step % self.config["train"]["log_steps"] == 0:
                save_logs(
                    dict(
                        epoch=self.postfix["Epoch"],
                        fold=self.postfix["Fold"],
                        step=step,
                        train_loss_step=f"{train_step['train_loss_step']:.3f}",
                        train_cls_loss_step=f"{train_step['train_cls_loss']:.3f}",
                        train_sim_loss_step=f"{train_step['train_sim_loss']:.3f}",
                        train_sim_test_loss_step=f"{train_step['train_sim_test_loss']:.3f}",
                    ),
                    output_dir=self.run_dir,
                    name="log_steps",
                )

        train_loss = torch.tensor(train_loss_list)
        train_cls_loss = torch.tensor(train_cls_loss_list)
        train_sim_loss = torch.tensor(train_sim_loss_list)
        train_sim_test_loss = torch.tensor(train_sim_test_loss)

        self.postfix["train_loss"] = train_loss.mean().item()
        self.postfix["train_cls_loss"] = train_cls_loss.mean().item()
        self.postfix["train_sim_loss"] = train_sim_loss.mean().item()
        self.postfix["train_sim_test_loss"] = train_sim_test_loss.mean().item()
        
        self.validation_procedure(model, optimizer, val_loader, epoch)
        self.overfit_check()     

    def training_step(self, model, optimizer, batch: BatchDict, t_loader, epoch, iteration) -> Dict[str, float]:
        with torch.autocast(
            device_type=self.config["device"].split(":")[0], enabled=self.config["train"]["mixed_precision"]
        ):

            anchor = model.forward(batch["anchor"].to(self.config["device"]), batch["anchor_label"].to(self.config["device"]), randomized=self.config["train"].get("randomized", False))
            positive = model.forward(batch["positive"].to(self.config["device"]), batch["positive_label"].to(self.config["device"]), randomized=self.config["train"].get("randomized", False))
            negative = model.forward(batch["negative"].to(self.config["device"]), batch["negative_label"].to(self.config["device"]), randomized=self.config["train"].get("randomized", False))

            outputs = torch.concatenate([anchor["cls"], positive["cls"], negative["cls"]])
            embeddings = torch.concatenate([anchor["f_c"], positive["f_c"], negative["f_c"]])
            labels = torch.concatenate([batch["anchor_label"], batch["anchor_label"], batch["negative_label"]]).long().to(self.config["device"])

            ntxent_loss = self.ntxent_loss(embeddings, labels)
            focal_loss = self.focal_loss(outputs, labels)
            loss = focal_loss + ntxent_loss 
            
            ntxent_with_test_loss = torch.tensor(-1).float()
    
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return {"train_loss_step": loss.item(), "train_cls_loss": focal_loss.item(), "train_sim_loss": ntxent_loss.item(), "train_sim_test_loss": ntxent_with_test_loss.item()}

    def validation_procedure(self, model, optimizer, val_loader, epoch=0) -> None:
        model.eval()
        embeddings: Dict[int, torch.Tensor] = {}
        for batch in tqdm(val_loader, total=len(val_loader), position=2, leave=False):
            val_dict = self.validation_step(model, batch)
            if val_dict["f_c"].ndim == 1:
                val_dict["f_c"] = val_dict["f_c"].unsqueeze(0)
            for anchor_id, embedding in zip(val_dict["anchor_id"], val_dict["f_c"]):
                embeddings[anchor_id] = embedding

        self.validation_epoch_end(embeddings, val_loader)
        save_logs(self.postfix, output_dir=self.run_dir)
        log.info(
            f"\n{' Validation Results ':=^50}\n"
            + "\n".join([f'"{key}": {value}' for key, value in self.postfix.items()])
            + f"\n{' End of Validation ':=^50}\n"
        )        
        model.train()

    def validation_step(self, model, batch: BatchDict) -> ValDict:
        anchor_id = batch["anchor_id"]
        features = model.forward(batch["anchor"].to(self.config["device"]))
        return {
            "anchor_id": anchor_id.numpy(),
            "f_c": features["f_c"].squeeze(0).detach().cpu(),
        }

    def validation_epoch_end(self, outputs: Dict[int, torch.Tensor], val_loader) -> Dict[int, np.ndarray]:
        clique_ids = []
        trackids = []
        for k, (anchor_id, embeddings) in enumerate(outputs.items()):
            clique_id = val_loader.dataset.version2clique.loc[anchor_id, 'clique']
            clique_ids.append(clique_id)
            trackids.append(anchor_id)
        preds = torch.stack(list(outputs.values()))
        rranks, average_precisions = calculate_ranking_metrics(embeddings=preds.numpy(), cliques=clique_ids)

        predictions = []
        for chunk_result in pairwise_distances_chunked(preds.numpy(), metric='cosine', reduce_func=reduce_func, working_memory=100):
            for query_indx, query_nearest_items in chunk_result:
                predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
                
        ndcg = calculate_ndcg(predictions, val_loader)
        self.postfix["mrr"] = rranks.mean()
        self.postfix["mAP"] = average_precisions.mean()
        self.postfix["nDCG"] = ndcg
        return {
            "rranks": rranks,
            "average_precisions": average_precisions,
        }

    def overfit_check(self) -> None:
        if self.early_stop(self.postfix["nDCG"]):
            log.info(f"\nValidation not improved for {self.early_stop.patience} consecutive epochs. Stopping...")
            self.state = "early_stopping"

        if self.early_stop.counter > 0:
            log.info("\nValidation nDCG was not improved")
        else:
            log.info(f"\nMetric improved. New best score: {self.early_stop.max_validation_mAP:.3f}")
            save_best_log(self.postfix, output_dir=self.run_dir)
            log.info("Saving model...")
            epoch = self.postfix["Epoch"]
            fold = self.postfix["Fold"]
            prev_model = deepcopy(self.best_model_path)
            self.best_model_path = os.path.join(
                self.run_dir, "model", f"best-model-{epoch=}-fold={fold}.pt"
            )
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(deepcopy(self.model.state_dict()), self.best_model_path)
            if prev_model is not None:
                os.remove(prev_model)
            self.folds_best_model_path[self.postfix["Fold"]] = self.best_model_path

    def test_model(self, fold) -> None:
        model = self.model_init(self.config)
        path = self.folds_best_model_path.get(fold, None)
        if path is not None:
            model.load_state_dict(torch.load(path), strict=False)
            log.info(f"Best model for fold {fold} loaded from checkpoint: {path}")
        else:
            raise ValueError

        self.test_procedure(model)

    def test_procedure(self, model) -> None:
        model.eval()
        trackids: List[int] = []
        embeddings: List[np.array] = []
        loader = copy.deepcopy(self.test_loader)
        for batch in tqdm(loader):
            test_dict = self.validation_step(model, batch)
            if test_dict["f_c"].ndim == 1:
                test_dict["f_c"] = test_dict["f_c"].unsqueeze(0)
            for anchor_id, embedding in zip(test_dict["anchor_id"], test_dict["f_c"]):
                trackids.append(anchor_id)
                embeddings.append(embedding.numpy())

        predictions_500 = []
        for chunk_result in pairwise_distances_chunked(embeddings, metric='cosine', reduce_func=reduce_func_500, working_memory=100):
            for query_indx, query_nearest_items in chunk_result:
                predictions_500.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
        save_test_predictions(predictions_500, output_dir=self.run_dir, fold=self.postfix["Fold"])
        self.predictions[self.postfix["Fold"]] = predictions_500

    def configure_optimizers(self, model) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=self.config["train"]["learning_rate"], weight_decay=self.config["train"]["weight_decay"])    

        return optimizer

    def average_predictions(self):
        averaged_predictions = []
        for prediction in range(len(self.predictions[0])):
            final_weight = {}
            for fold in range(len(self.predictions)):
                weights = {k:1/np.sqrt(i+1) for i,k in enumerate(np.array(self.predictions[fold][prediction][1]))}
                for key in list(weights.keys()):
                    final_weight[key] = final_weight.get(key, 0) + weights.get(key, 0)

            new_order = [k for k, v in sorted(final_weight.items(), key=lambda item: -item[1])][:100]
            averaged_predictions.append((self.predictions[fold][prediction][0], new_order))
        save_test_predictions(averaged_predictions, output_dir=self.run_dir)