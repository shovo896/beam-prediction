import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularBeamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.act(x + self.drop(self.block(x)))


class BeamTabularNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, depth=4, dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def topk_accuracies(logits, targets, ks=(1, 3, 5)):
    max_k = max(ks)
    _, pred = torch.topk(logits, k=max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        out[k] = correct[:k].reshape(-1).float().sum().item() * 100.0 / targets.size(0)
    return out


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    sums = {1: 0.0, 3: 0.0, 5: 0.0}

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total += bs

            accs = topk_accuracies(logits, y)
            for k in sums:
                sums[k] += accs[k] * bs

    return {
        "loss": total_loss / total,
        "top1": sums[1] / total,
        "top3": sums[3] / total,
        "top5": sums[5] / total,
    }


@dataclass
class TabularPrep:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    imputer: SimpleImputer


def prepare_tabular_data_v2(train_df, test_df, feature_cols, label_col="label_id", add_feature_engineering=True):
    train_df = train_df.copy()
    test_df = test_df.copy()

    use_cols = list(feature_cols)

    for df in (train_df, test_df):
        for col in use_cols + [label_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if add_feature_engineering:
        if {"lat", "lon"}.issubset(use_cols):
            for df in (train_df, test_df):
                df["lat_lon_radius"] = np.sqrt(df["lat"] ** 2 + df["lon"] ** 2)
                df["lat_lon_angle"] = np.arctan2(df["lon"], df["lat"])
                df["lat_x_lon"] = df["lat"] * df["lon"]
                df["lat_minus_lon"] = df["lat"] - df["lon"]
            use_cols += ["lat_lon_radius", "lat_lon_angle", "lat_x_lon", "lat_minus_lon"]

        if "unit2_height" in use_cols:
            for df in (train_df, test_df):
                df["height_sq"] = df["unit2_height"] ** 2
            use_cols += ["height_sq"]

        if "unit2_distance" in use_cols:
            for df in (train_df, test_df):
                df["distance_sq"] = df["unit2_distance"] ** 2
                df["inv_distance"] = 1.0 / (df["unit2_distance"].abs() + 1e-6)
            use_cols += ["distance_sq", "inv_distance"]

        if {"unit2_height", "unit2_distance"}.issubset(use_cols):
            for df in (train_df, test_df):
                df["height_distance_ratio"] = df["unit2_height"] / (df["unit2_distance"].abs() + 1e-6)
                df["height_x_distance"] = df["unit2_height"] * df["unit2_distance"]
            use_cols += ["height_distance_ratio", "height_x_distance"]

    train_df = train_df.dropna(subset=[label_col]).copy()
    test_df = test_df.dropna(subset=[label_col]).copy()

    y_train = train_df[label_col].astype(np.int64).values
    y_test = test_df[label_col].astype(np.int64).values

    X_train = train_df[use_cols].values.astype(np.float32)
    X_test = test_df[use_cols].values.astype(np.float32)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test = imputer.transform(X_test).astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return TabularPrep(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        imputer=imputer,
    )


def make_weighted_sampler(y):
    class_counts = np.bincount(y)
    class_counts[class_counts == 0] = 1
    sample_weights = 1.0 / class_counts[y]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_model(
    train_loader,
    valid_loader,
    input_dim,
    num_classes,
    epochs=150,
    lr=2e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    patience=25,
):
    model = BeamTabularNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        depth=4,
        dropout=0.15,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            total += bs

        scheduler.step()

        metrics = evaluate_model(model, valid_loader, criterion, device)
        train_loss = running_loss / max(total, 1)
        score = metrics["top1"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": metrics["loss"],
                "top1": metrics["top1"],
                "top3": metrics["top3"],
                "top5": metrics["top5"],
            }
        )

        if score > best_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Train {train_loss:.4f} | Valid {metrics['loss']:.4f} | "
                f"Top1 {metrics['top1']:.2f} | Top3 {metrics['top3']:.2f} | Top5 {metrics['top5']:.2f}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}, best Top1 {best_metric:.2f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch, best_metric


def cross_validate_tabular(
    train_df,
    feature_cols,
    num_classes,
    label_col="label_id",
    n_splits=5,
    batch_size=128,
    seed=42,
):
    y = pd.to_numeric(train_df[label_col], errors="coerce").astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_models = []
    fold_preps = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), start=1):
        fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[va_idx].reset_index(drop=True)

        prep = prepare_tabular_data_v2(fold_train, fold_valid, feature_cols, label_col=label_col)
        train_ds = TabularBeamDataset(prep.X_train, prep.y_train)
        valid_ds = TabularBeamDataset(prep.X_test, prep.y_test)

        sampler = make_weighted_sampler(prep.y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model, hist, best_epoch, best_metric = train_one_model(
            train_loader=train_loader,
            valid_loader=valid_loader,
            input_dim=prep.X_train.shape[1],
            num_classes=num_classes,
        )

        print(f"Fold {fold}: best epoch {best_epoch}, best Top1 {best_metric:.2f}")
        fold_models.append(model)
        fold_preps.append(prep)
        fold_scores.append(best_metric)

    print(f"CV Top1 mean={np.mean(fold_scores):.2f}, std={np.std(fold_scores):.2f}")
    return fold_models, fold_preps, fold_scores


def predict_with_ensemble(models, preps, test_df, feature_cols):
    probs_sum = None
    y_true = None

    for model, prep in zip(models, preps):
        temp = test_df.copy()

        use_cols = list(feature_cols)
        if {"lat", "lon"}.issubset(use_cols):
            temp["lat_lon_radius"] = np.sqrt(temp["lat"] ** 2 + temp["lon"] ** 2)
            temp["lat_lon_angle"] = np.arctan2(temp["lon"], temp["lat"])
            temp["lat_x_lon"] = temp["lat"] * temp["lon"]
            temp["lat_minus_lon"] = temp["lat"] - temp["lon"]
            use_cols += ["lat_lon_radius", "lat_lon_angle", "lat_x_lon", "lat_minus_lon"]

        if "unit2_height" in use_cols:
            temp["height_sq"] = temp["unit2_height"] ** 2
            use_cols += ["height_sq"]

        if "unit2_distance" in use_cols:
            temp["distance_sq"] = temp["unit2_distance"] ** 2
            temp["inv_distance"] = 1.0 / (temp["unit2_distance"].abs() + 1e-6)
            use_cols += ["distance_sq", "inv_distance"]

        if {"unit2_height", "unit2_distance"}.issubset(use_cols):
            temp["height_distance_ratio"] = temp["unit2_height"] / (temp["unit2_distance"].abs() + 1e-6)
            temp["height_x_distance"] = temp["unit2_height"] * temp["unit2_distance"]
            use_cols += ["height_distance_ratio", "height_x_distance"]

        X = temp[use_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        X = prep.imputer.transform(X).astype(np.float32)
        X = prep.scaler.transform(X).astype(np.float32)
        y_true = temp["label_id"].astype(int).values

        loader = DataLoader(TabularBeamDataset(X, y_true), batch_size=256, shuffle=False)
        probs = []

        model.eval()
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs

    probs_mean = probs_sum / len(models)
    pred = np.argmax(probs_mean, axis=1)

    top1 = accuracy_score(y_true, pred) * 100.0

    topk_metrics = {}
    topk_idx = np.argsort(-probs_mean, axis=1)
    for k in [1, 3, 5]:
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] in topk_idx[i, :k]:
                correct += 1
        topk_metrics[k] = 100.0 * correct / len(y_true)

    return {
        "top1": topk_metrics[1],
        "top3": topk_metrics[3],
        "top5": topk_metrics[5],
        "pred": pred,
        "y_true": y_true,
    }


def run_improved_tabular_experiment(train_df, test_df, feature_cols, num_classes, name):
    print(f"\n===== {name} =====")
    models, preps, cv_scores = cross_validate_tabular(
        train_df=train_df,
        feature_cols=feature_cols,
        num_classes=num_classes,
        label_col="label_id",
        n_splits=5,
        batch_size=128,
        seed=42,
    )
    metrics = predict_with_ensemble(models, preps, test_df, feature_cols)
    print(
        f"{name} Test Top1={metrics['top1']:.2f} | "
        f"Top3={metrics['top3']:.2f} | Top5={metrics['top5']:.2f}"
    )
    return {
        "name": name,
        "cv_top1_mean": float(np.mean(cv_scores)),
        "cv_top1_std": float(np.std(cv_scores)),
        "test_top1": metrics["top1"],
        "test_top3": metrics["top3"],
        "test_top5": metrics["top5"],
        "models": models,
        "preps": preps,
    }


# Paste and run this part inside the notebook after label_id is created.
"""
set_seed(42)

results_pos_h = run_improved_tabular_experiment(
    train_df=train_pos_h,
    test_df=test_pos_h,
    feature_cols=["lat", "lon", "unit2_height"],
    num_classes=num_classes,
    name="Position + Height"
)

results_pos_h_d = run_improved_tabular_experiment(
    train_df=train_pos_h_d,
    test_df=test_pos_h_d,
    feature_cols=["lat", "lon", "unit2_height", "unit2_distance"],
    num_classes=num_classes,
    name="Position + Height + Distance"
)

summary = pd.DataFrame([
    {
        "model": results_pos_h["name"],
        "cv_top1_mean": results_pos_h["cv_top1_mean"],
        "cv_top1_std": results_pos_h["cv_top1_std"],
        "test_top1": results_pos_h["test_top1"],
        "test_top3": results_pos_h["test_top3"],
        "test_top5": results_pos_h["test_top5"],
    },
    {
        "model": results_pos_h_d["name"],
        "cv_top1_mean": results_pos_h_d["cv_top1_mean"],
        "cv_top1_std": results_pos_h_d["cv_top1_std"],
        "test_top1": results_pos_h_d["test_top1"],
        "test_top3": results_pos_h_d["test_top3"],
        "test_top5": results_pos_h_d["test_top5"],
    },
])

summary
"""
