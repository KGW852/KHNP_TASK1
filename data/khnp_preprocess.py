# data/khnp_preprocess.py

import os
import random

from configs.config import Config
from utils.csv_utils import read_csv, read_second_column
from data.tools_preprocess import Segmenter, NoiseData, BaselineFitter

from math import ceil
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def get_configs():
    config = Config.exp1()
    cfg = config.config_dict
    return cfg

def _collect_paths(d):
    paths = []
    for v in d.values():
        if isinstance(v, dict):
            paths.extend(_collect_paths(v))
        elif isinstance(v, str):
            paths.append(v)
    return paths
    
class Preprocessor:
    """khnp Acc-csv file to STFT-png & train/eval/test set"""
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # original path
        orig = cfg["orig_dir"]
        self.orig_s0 = orig["s0_dir"]
        self.orig_s1 = orig["s1_dir"]
        self.orig_s2 = orig["s2_dir"]
        self.orig_s3 = orig["s3_dir"]
        self.orig_s4 = orig["s4_dir"]
        
        # save path
        self.train_dirs = cfg["train_dir"]
        self.eval_dirs  = cfg["eval_dir"]
        self.test_dirs  = cfg["test_dir"]
        for p in _collect_paths(self.train_dirs) + _collect_paths(self.eval_dirs) + _collect_paths(self.test_dirs):
            os.makedirs(p, exist_ok=True)

        # noise param
        noise = cfg["noise"]
        self.save_dir = noise["save_dir"]
        self.src_scale = noise["src_scale"]
        self.tgt_scale = noise["tgt_scale"]
        self.src_modulation = noise["src_modulation"]
        self.tgt_modulation = noise["tgt_modulation"]
        self.src_iter = noise["src_iter"]
        self.tgt_iter = noise["tgt_iter"]

        # preprocess param
        pp = cfg["preprocess"]
        self.random_seed = pp["random_seed"]
        self.sr = pp["sr"]
        self.nsample = pp["nsample"]
        self.n_fft = pp["n_fft"]
        self.hop = pp["hop_length"]
        self.window = pp["window"]
        self.nsplit = pp["nsplit"]

        # utils
        self.segmenter = Segmenter(self.nsample)
        self.baseline = BaselineFitter()
        self.noise_data = NoiseData(self.save_dir)

    def _sig2spec(self, sig: np.ndarray):
        """Calculate STFT"""
        spec = librosa.stft(sig, n_fft=self.n_fft,
                            hop_length=self.hop, window=self.window, center=False)
        s_mag = np.abs(spec).astype(np.float32)
        return s_mag

    def _save_png(self, spec: np.ndarray, save_path: str):
        """Save PNG"""
        # spec[0, :]: low frequency at the bottom
        spec_vis = np.flipud(spec)
        
        vmin, vmax = float(spec_vis.min()), float(spec_vis.max())
        plt.imsave(save_path, spec_vis, cmap='magma', vmin=vmin, vmax=vmax)

    def _process_csv(self, csv_path: str, domain: str): 
        """Processing CSV File"""
        rows = read_csv(csv_path, skip_header=True)
        raw = np.asarray(read_second_column(rows), dtype=np.float32)
        if raw.size == 0:
            return []
        
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        specs = []

        raw_corr = self.baseline.fit_correct(raw)  # baseline

        for seg_idx, (start, end) in enumerate(self.segmenter.data_overlap(len(raw_corr))):
            seg = raw_corr[start:end]
            seg = self.segmenter.data_hanning(seg)  # hanning

            spec = self._sig2spec(seg)  # STFT
            tag = f"{fname}_{seg_idx}"
            specs.append((spec, tag, domain))  # list[tuple[np.ndarray, str, str]]:  # (spec, tag, domain)

        return specs

    def _process_dir(self, dir_key: str, domain: str):
        """Processing CSV Files in Directory"""
        csv_dir_map = {
            "s0_dir": self.orig_s0,
            "s1_dir": self.orig_s1,
            "s2_dir": self.orig_s2,
            "s3_dir": self.orig_s3,
            "s4_dir": self.orig_s4
        }
        csv_dir = csv_dir_map.get(dir_key)
        if not csv_dir or not os.path.isdir(csv_dir):
            return []
        
        out = []  # list[tuple[np.ndarray, str, str]]:
        for fn in os.listdir(csv_dir):
            if fn.lower().endswith(".csv"):
                out.extend(self._process_csv(csv_path=os.path.join(csv_dir, fn), domain=domain))

        return out

    def _split_and_save(self, src_specs):
        """Segment and save data"""
        bucket = defaultdict(list)
        for spec, tag, dom in src_specs:
            bucket[dom].append((spec, tag))
        
        # domain
        s0s = bucket["s0"]
        s1s = bucket["s1"]
        s2s = bucket["s2"]
        s3s = bucket["s3"]
        s4s = bucket["s4"]

        # normal to train, eval00
        if isinstance(self.nsplit, float):  # ratio
            train_norm, eval_norm = train_test_split(s0s, test_size=self.nsplit, random_state=self.random_seed, shuffle=True)
        else:  # absolute
            k = min(self.nsplit, len(s0s))
            train_norm, eval_norm = train_test_split(s0s, test_size=k, random_state=self.random_seed, shuffle=True)
        
        # eval01, eval02, eval03, eval04: sampling train_norm as s1·s2·s3·s4
        def pick_normals(src_pool, k):
            if k <= 0 or not src_pool:
                return []
            if k <= len(src_pool):
                return random.sample(src_pool, k)
            return random.choices(src_pool, k=k)

        eval01_norm = pick_normals(eval_norm, len(s1s))
        eval02_norm = pick_normals(eval_norm, len(s2s))
        eval03_norm = pick_normals(eval_norm, len(s3s))
        eval04_norm = pick_normals(eval_norm, len(s4s))

        # save helper
        def dump(specs, root):
            os.makedirs(root, exist_ok=True)
            for spec, tag in specs:
                self._save_png(spec, os.path.join(root, f"{tag}.png"))

        # save
        dump(train_norm, self.train_dirs["train"])

        dump(eval_norm, self.eval_dirs["eval00"])

        dump(eval01_norm, self.test_dirs["eval01"])
        dump(eval02_norm, self.test_dirs["eval02"])
        dump(eval03_norm, self.test_dirs["eval03"])
        dump(eval04_norm, self.test_dirs["eval04"])
        dump(s1s, self.test_dirs["eval01"])
        dump(s2s, self.test_dirs["eval02"])
        dump(s3s, self.test_dirs["eval03"])
        dump(s4s, self.test_dirs["eval04"])

    def run(self):
        """Run preprocessing pipeline"""
        # augmentation & STFT
        src_specs = []
        for step_idx, k in enumerate(('s0_dir', 's1_dir', 's2_dir', 's3_dir', 's4_dir'), start=1):
            domain = k.split('_')[0]
            src_specs += self._process_dir(dir_key=k, domain=domain)
            print(f"[{step_idx}/6] SRC {k} data processed. ✅")

        # split → save png
        self._split_and_save(src_specs)
        print(f"[6/6] SRC {len(src_specs)} seg processed. ✅")

if __name__ == "__main__":
    cfg = get_configs()
    Preprocessor(cfg).run()
