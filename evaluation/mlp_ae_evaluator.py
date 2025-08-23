# evaluation/mlp_ae_evaluator.py

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.mlp_ae import MLPAE
from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from utils.datalist_utils import remove_duplicates
from evaluation.modules.anomaly_metrics import AnomalyScore, AnomalyMetric
from evaluation.modules.umap import UMAPPlot
from evaluation.modules.pca import PCAPlot
from evaluation.modules.histogram import HistPlot
from evaluation.modules.all_scores import ExtractScore


class MLPAEEvaluator:
    """
    Simple MLP AutoEncoder evaluator
    Args:
        cfg (dict): config dictionary
        mlflow_logger (MLFlowLogger): MLflow logger(define param in 'main.py')
        device (torch.device): cuda or cpu
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        self.model = MLPAE(
            channels=cfg["mlp_ae"]["channels"],
            height=cfg["mlp_ae"]["height"],
            width=cfg["mlp_ae"]["width"],
            enc_hidden_dims=cfg["mlp_ae"]["enc_hidden_dims"],
            enc_latent_dim=cfg["mlp_ae"]["enc_latent_dim"],
            dec_latent_dim=cfg["mlp_ae"]["dec_latent_dim"],
            dec_hidden_dims=cfg["mlp_ae"]["dec_hidden_dims"],
            ae_dropout=cfg["mlp_ae"]["dropout"],
            ae_use_batchnorm=cfg["mlp_ae"]["use_batch_norm"]
        ).to(self.device)

        # run param
        self.eval_run_epoch = cfg["eval"]["epoch"]
        self.test_run_epoch = cfg["test"]["epoch"]

        # utils: model manage, mlflow
        self.model_utils = ModelUtils(self.cfg)
        self.mlflow_logger = mlflow_logger

        # run_name: model_name
        self.run_name = self.model_utils.get_model_name()

        # Metrics modules
        self.anomaly_score = AnomalyScore(self.cfg)
        self.umap = UMAPPlot(self.cfg)
        self.pca = PCAPlot(self.cfg)
        self.extract_score = ExtractScore(self.cfg)
        self.method = cfg["anomaly"]["method"]

    def load_checkpoint(self, epoch: int):
        file_name = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(file_name)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Cannot find checkpoint file: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def test_epoch(self, data_loader, epoch):
        self.load_checkpoint(epoch)
        self.model.eval()

        src_results= []

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[Test]  [Epoch {epoch}] | Metric: {self.method}", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                src_data, src_label, src_path = data  # src_label, tgt_label: tensor([class_label, anomaly_label])
                x_s = src_data.to(self.device)

                # forward
                (e_s, x_s_recon) = self.model(x_s)

                batch_size = len(src_path)
                for i in range(batch_size):
                    class_label_src = src_label[i][0].item()
                    anomaly_label_src = src_label[i][1].item()

                    # results
                    src_results.append({
                        "file_name"     : src_path[i],
                        "x"             : x_s[i].cpu().numpy(),
                        "encoder"       : e_s[i].cpu().numpy(),
                        "x_recon"       : x_s_recon[i].cpu().numpy(),
                        "class_label"   : class_label_src,
                        "anomaly_label" : anomaly_label_src,
                        "domain"        : "src"
                    })

                    src_results = remove_duplicates(src_results, key_name="file_name")

        return src_results

    def run(self, eval_loader, test_loader):
        """Evaluating MLflow run: Load the epoch model"""
        if self.mlflow_logger is not None:
            run_id_path = self.model_utils.get_file_path(f"{self.run_name}.json")
            run_id, exp_id, exp_name = self.mlflow_logger.load_run_id(run_id_path)
            self.mlflow_logger.resume_run(run_id)

        test_run_epochs = self.model_utils.get_run_epochs(self.test_run_epoch)

        for epoch in test_run_epochs:
            test_results = self.test_epoch(test_loader, epoch)

            # Latent space visualization
            enc_list = [res["encoder"] for res in test_results]
            class_labels = [res["class_label"] for res in test_results]
            anomaly_labels = [res["anomaly_label"] for res in test_results]

            save_dir = self.model_utils.get_save_dir()
            os.makedirs(f"{save_dir}/umap", exist_ok=True)
            enc_umap_path = f"{save_dir}/umap/umap_s1_encoder_epoch{epoch}.png"
            enc_umap_data_path = f"{save_dir}/umap/umap_s1_encoder_epoch{epoch}.csv"

            os.makedirs(f"{save_dir}/pca", exist_ok=True)
            enc_pca_path = f"{save_dir}/pca/pca_s1_encoder_epoch{epoch}.png"

            enc_np = np.stack(enc_list, axis=0)
            class_np = np.array(class_labels)
            anomaly_np = np.array(anomaly_labels)

            self.umap.plot_umap(
                save_path=enc_umap_path,
                csv_path = enc_umap_data_path,
                features=enc_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=None,
                radius=None,
                boundary_samples=None
            )
            self.pca.plot_pca(
                save_path=enc_pca_path,
                features=enc_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=None,
                radius=None,
                boundary_samples=None
            )

            # Anomaly detection performance evaluation
            file_names, y_true, y_scores = [], [], []
            for src_res in test_results:
                file_name = src_res["file_name"]
                anomaly_label = src_res["anomaly_label"]

                x = torch.tensor(src_res["x"]).unsqueeze(0).to(self.device)
                x_recon = torch.tensor(src_res["x_recon"]).unsqueeze(0).to(self.device)
                score_tensor = self.anomaly_score.anomaly_score(x_recon=x_recon, x=x)

                file_names.append(file_name)
                y_true.append(anomaly_label)
                y_scores.append(score_tensor.item())

            os.makedirs(f"{save_dir}/metric", exist_ok=True)
            anomaly_data_path = f"{save_dir}/metric/anomaly_s1_scores_epoch{epoch}_{self.method}.csv"
            anomaly_metric_path = f"{save_dir}/metric/anomaly_s1_metric_epoch{epoch}_{self.method}.csv"

            # Histogram plot
            #hist_scores = [res["distance"] for res in test_results]

            os.makedirs(f"{save_dir}/histogram", exist_ok=True)
            hist_path = f"{save_dir}/histogram/anomaly_s1_hist_epoch{epoch}_{self.method}.png"

            # Extract all scores
            os.makedirs(f"{save_dir}/scores", exist_ok=True)
            recon_scores_path = f"{save_dir}/scores/s1_scores_epoch{epoch}_recon.csv"
            #distance_scores_path = f"{save_dir}/scores/s1_scores_epoch{epoch}_distance.csv"

            # Results
            anomaly_metric = AnomalyMetric(cfg=self.cfg, file_name=file_names, y_true=y_true, y_score=y_scores)
            anomaly_dict = anomaly_metric.calc_metric()
            print(f"[Test]  [Epoch {epoch}] | Metric: {self.method} | ", anomaly_dict)

            anomaly_metric.save_anomaly_scores_as_csv(data_csv_path=anomaly_data_path, metric_csv_path=anomaly_metric_path)

            hist_plot = HistPlot(anomaly_dict=anomaly_dict)
            hist_plot.plot_hist(save_path=hist_path, scores=y_scores, class_labels=class_labels)

            self.extract_score.recon_scores(csv_path=recon_scores_path, results_list=test_results)
            #self.extract_score.distance_scores(csv_path=distance_scores_path, results_list=test_results)

            # mlflow
            if self.mlflow_logger:
                self.mlflow_logger.log_artifact(enc_umap_path, artifact_path="umap")
                self.mlflow_logger.log_artifact(enc_umap_data_path, artifact_path="umap")
                self.mlflow_logger.log_artifact(enc_pca_path, artifact_path="pca")
                self.mlflow_logger.log_artifact(anomaly_data_path, artifact_path="metrics")
                self.mlflow_logger.log_artifact(anomaly_metric_path, artifact_path="metrics")
                self.mlflow_logger.log_artifact(hist_path, artifact_path="histogram")
                self.mlflow_logger.log_artifact(recon_scores_path, artifact_path="scores")
                #self.mlflow_logger.log_artifact(distance_scores_path, artifact_path="scores")

        if self.mlflow_logger:
            self.mlflow_logger.end_run()
