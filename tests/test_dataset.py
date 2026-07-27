import pytest
from PIL import Image

from dataset import ROPDataset, stratified_kfold_splits, stratified_split

FOLDER_COUNTS = {
    "Neo_Normal": 4,
    "RetCam_Normal": 4,
    "Neo_ROP": 4,
    "RetCam_ROP": 4,
}


@pytest.fixture
def dataset_root(tmp_path):
    for folder, count in FOLDER_COUNTS.items():
        folder_path = tmp_path / folder
        folder_path.mkdir()
        for i in range(count):
            Image.new("RGB", (8, 8), color=(i, i, i)).save(folder_path / f"img_{i}.png")
    return str(tmp_path)


def test_ropdataset_counts_and_labels(dataset_root):
    ds = ROPDataset(dataset_root)

    assert len(ds) == sum(FOLDER_COUNTS.values())
    assert ds.labels.count(0) == FOLDER_COUNTS["Neo_Normal"] + FOLDER_COUNTS["RetCam_Normal"]
    assert ds.labels.count(1) == FOLDER_COUNTS["Neo_ROP"] + FOLDER_COUNTS["RetCam_ROP"]


def test_stratified_split_is_disjoint_and_covers_all(dataset_root):
    ds = ROPDataset(dataset_root)
    all_indices = list(range(len(ds)))

    keep, holdout = stratified_split(all_indices, ds.labels, holdout_size=0.25, seed=42)

    assert set(keep).isdisjoint(set(holdout))
    assert set(keep) | set(holdout) == set(all_indices)
    # stratification should give holdout at least one sample per class
    holdout_labels = {ds.labels[i] for i in holdout}
    assert holdout_labels == {0, 1}


def test_kfold_splits_are_disjoint_and_cover_pool(dataset_root):
    ds = ROPDataset(dataset_root)
    all_indices = list(range(len(ds)))
    trainval_idx, _test_idx = stratified_split(all_indices, ds.labels, holdout_size=0.25, seed=42)

    n_folds = 4
    folds = stratified_kfold_splits(trainval_idx, ds.labels, n_folds=n_folds, seed=42)
    assert len(folds) == n_folds

    val_indices_seen = []
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == set(trainval_idx)
        val_indices_seen.extend(val_idx)

    # every pool index shows up in exactly one fold's validation split
    assert sorted(val_indices_seen) == sorted(trainval_idx)
