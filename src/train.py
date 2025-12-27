from __future__ import annotations

"""
Module d'entraînement et d'enregistrement d'un modèle de churn.

Ce script :
1) Charge `data/processed.csv`
2) Sépare X/y (cible: churn)
3) Construit un pipeline scikit-learn (prétraitement + LogisticRegression)
4) Split train/test
5) Entraîne et évalue (accuracy, precision, recall, F1)
6) Compare la F1 à une baseline (prédire toujours 0) + un seuil gate
7) Sauvegarde :
   - un modèle horodaté dans models/
   - les métadonnées dans registry/metadata.json
   - si gate OK :
       * écrit registry/current_model.txt
       * écrit un alias stable models/model.joblib (pour DVC evaluate)

But MLOps : traçabilité + "model registry" minimal avec un gate qualité.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Final

import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Chemins et constantes globales
# ---------------------------------------------------------------------------

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODELS_DIR: Final[Path] = ROOT / "models"
REGISTRY_DIR: Final[Path] = ROOT / "registry"

CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Fonctions pour la gestion des métadonnées
# ---------------------------------------------------------------------------


def load_metadata() -> list[dict[str, Any]]:
    """
    Charge la liste des métadonnées de modèles depuis METADATA_PATH.
    Si le fichier n'existe pas, retourne une liste vide.
    """
    if not METADATA_PATH.exists():
        return []
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(items: list[dict[str, Any]]) -> None:
    """
    Sauvegarde la liste des métadonnées de modèles dans METADATA_PATH.
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------


def compute_baseline_f1(y_true: pd.Series | list[int]) -> float:
    """
    Baseline : prédire toujours 0 (pas de churn).
    Retourne la F1 correspondante.
    """
    y_pred = [0] * len(y_true)
    return float(f1_score(y_true, y_pred, zero_division=0))


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Préprocessing :
    - num : StandardScaler
    - cat : OneHotEncoder(handle_unknown="ignore")
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_model_pipeline(preprocessor: ColumnTransformer, seed: int) -> Pipeline:
    """
    Pipeline complet : préprocesseur + régression logistique.
    """
    clf = LogisticRegression(
        max_iter=200,
        random_state=seed,
    )
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------


def main(version: str = "v1", seed: int = 42, gate_f1: float = 0.70) -> None:
    """
    Entraîne et enregistre un modèle de churn.

    Params
    ------
    version : identifiant logique (ex: v1, v2)
    seed : graine reproductible
    gate_f1 : seuil minimal F1
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Fichier data/processed.csv introuvable. "
            "Exécute d'abord la préparation des données."
        )

    # Chargement des données
    df = pd.read_csv(DATA_PATH)

    target_col = "churn"
    if target_col not in df.columns:
        raise ValueError(
            "Colonne cible 'churn' introuvable dans processed.csv. "
            "Vérifie la préparation des données."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Colonnes attendues (adapte si besoin selon ton dataset)
    numeric_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]
    categorical_cols = ["plan_type", "region"]

    # Vérification simple : colonnes présentes
    missing_cols = [c for c in (numeric_cols + categorical_cols) if c not in X.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans processed.csv: {missing_cols}. "
            "Adapte numeric_cols / categorical_cols."
        )

    # Construction pipeline
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    model_pipeline = build_model_pipeline(preprocessor, seed=seed)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    # Entraînement
    model_pipeline.fit(X_train, y_train)

    # Prédictions
    y_pred = model_pipeline.predict(X_test)

    # Métriques
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "baseline_f1": compute_baseline_f1(y_test),
    }

    # Sauvegarde modèle horodaté
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_filename = f"churn_model_{version}_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename
    joblib.dump(model_pipeline, model_path)
    # Alias stable pour DVC (evaluate dépend de ce fichier)
    stable_model_path = MODELS_DIR / "model.joblib"
    joblib.dump(model_pipeline, stable_model_path)
    print(f"[OK] Alias stable : {stable_model_path}")


    # Gate
    passed_gate = bool(metrics["f1"] >= gate_f1 and metrics["f1"] >= metrics["baseline_f1"])

    # Entrée metadata
    entry: dict[str, Any] = {
        "model_file": model_filename,
        "version": version,
        "trained_at_utc": timestamp,
        "data_file": DATA_PATH.name,
        "seed": seed,
        "metrics": metrics,
        "gate_f1": gate_f1,
        "passed_gate": passed_gate,
    }

    # Update metadata registry
    items = load_metadata()
    items.append(entry)
    save_metadata(items)

    # Logs
    print("[METRICS]", json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[OK] Modèle sauvegardé : {model_path}")

    # "Déploiement" minimal : pointer le modèle courant + alias stable
    if entry["passed_gate"]:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

        # écrit le nom du modèle courant (utile pour audit / traçabilité)
        CURRENT_MODEL_PATH.write_text(model_filename, encoding="utf-8")

        # alias stable pour DVC (evaluate dépendra toujours de models/model.joblib)
        stable_model_path = MODELS_DIR / "model.joblib"
        joblib.dump(model_pipeline, stable_model_path)
        # Alias stable pour DVC (evaluate dépend de ce fichier)
        stable_model_path = MODELS_DIR / "model.joblib"
        joblib.dump(model_pipeline, stable_model_path)
        print(f"[OK] Alias stable : {stable_model_path}")


        print(f"[DEPLOY] Modèle activé : {model_filename}")
        print(f"[DEPLOY] Alias stable : {stable_model_path}")
    else:
        print("[DEPLOY] Refusé : F1 insuffisante ou baseline non battue.")


if __name__ == "__main__":
    main()