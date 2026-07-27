from dataclasses import dataclass

import yaml


@dataclass
class Config:
    data_root: str
    output_dir: str

    image_size: int
    batch_size: int
    num_workers: int

    epochs_phase1: int
    epochs_phase2: int
    lr_phase1: float
    lr_phase2: float
    weight_decay: float

    test_size: float
    n_folds: int
    final_val_size: float

    seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
