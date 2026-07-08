# Data Separation & Splitting Framework

To maintain scientific integrity and prevent data leakage, we partition our database of 200 subjects into three distinct splits: **Train**, **Validation**, and **Test**. 

---

## 1. Split Partitioning & Seed

The partition proportions and configuration are fixed as follows:
* **Train split (70%)**: 140 subjects. Used to fit the statistical shape model (SSM) and train the Gradient Boosting Regressors (GBR) coordinate correction models.
* **Validation split (15%)**: 30 subjects. Used for hyperparameter tuning, model architecture selection, and general feedback during development.
* **Test split (15%)**: 30 subjects. Strictly reserved for final, one-shot reporting before challenge submission.

The splitting process uses a deterministic random seed:
* **`SPLIT_SEED = 42`**

### Why is the split frozen?
In statistical shape estimation, prediction algorithms often memorize or blend local details of known subject shapes (such as the KNN blending method `_knn_predict`). If the evaluation set overlaps with the training set, performance metrics will artificially reflect shape memorization instead of generalizability. 

Freezing these splits ensures that all models are evaluated on completely unseen anatomical shapes.

---

## 2. Core Rule of Evaluators
1. **Train on Train**: Only fit models using the `train` split.
2. **Tune on Validation**: Measure progress and tune hyperparameters on the `val` split.
3. **Report Test Once**: Evaluate on the `test` split once at the end of the project.

**Loud Leakage Guard**:
The evaluation script `evaluate.py` implements a validation check on startup. If it detects that any subject from the selected evaluation split was present in the model's training set, it halts execution immediately to prevent compromised validation scores.

---

## 3. Teammate CLI Guide

### How to regenerate the frozen splits
If you need to regenerate the split text files (e.g. if new subjects are added to the database), run:
```bash
python scripts/make_splits.py --mesh-dir "path/to/mesh"
```
This utility reads all subject PLY meshes, sorts their IDs alphabetically to ensure system-agnostic order, partitions them deterministically, and writes the frozen splits under the `splits/` directory:
* `splits/train_pids.txt`
* `splits/val_pids.txt`
* `splits/test_pids.txt`

### How to train only on the Train split
By default, `train.py` runs on the `train` split:
```bash
python train.py
```
To run final model training on both train and validation splits:
```bash
python train.py --include-val
```

### How to evaluate on the Validation split
By default, `evaluate.py` runs on the `val` split:
```bash
python evaluate.py
```
To evaluate on the test split for final reporting:
```bash
python evaluate.py --split test
```
