"""
FarrahAI — Module 6: ML Models
================================
All supervised and unsupervised ML algorithms live here.

Supervised (with labels):
  - Topic classification → Logistic Regression, XGBoost
  - Question type classification → Random Forest
  - Topic importance prediction → XGBoost

Unsupervised (no labels):
  - Topic clustering → K-Means, Hierarchical
  - Question grouping → DBSCAN

Metrics shown:
  - Accuracy, Precision, Recall, F1 (supervised)
  - Silhouette Score (unsupervised)
  - Confusion matrix
"""

import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Feature Extraction ────────────────────────────────────────────────────────

def build_tfidf_features(texts: list[str], max_features: int = 5000):
    """
    Build TF-IDF feature matrix from texts.
    Returns (vectorizer, feature_matrix)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vec.fit_transform(texts)
    return vec, X


def build_embedding_features(texts: list[str],
                              model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Use sentence embeddings as features for ML models.
    Often better than TF-IDF for semantic tasks.
    """
    from modules.embedder import embed_texts
    return embed_texts(texts, model_name=model_name)


# ── Supervised Learning ───────────────────────────────────────────────────────

def train_topic_classifier(X, y, model_type: str = "xgboost"):
    """
    Train a topic classifier.

    Args:
        X: feature matrix (TF-IDF or embeddings)
        y: topic labels
        model_type: "xgboost" | "logistic" | "random_forest"

    Returns:
        (trained_model, metrics_dict)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = _get_classifier(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model":     model_type,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average='macro'), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average='weighted'), 4),
        "report":    classification_report(y_test, y_pred,
                                           target_names=le.classes_),
        "label_encoder": le,
    }

    print(f"\n── {model_type.upper()} Topic Classifier ──────────────────")
    print(f"  Accuracy:    {metrics['accuracy']}")
    print(f"  F1 (macro):  {metrics['f1_macro']}")
    print(f"  F1 (weighted): {metrics['f1_weighted']}")
    print(f"\n{metrics['report']}")

    return model, metrics


def _get_classifier(model_type: str):
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    elif model_type == "svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(max_iter=2000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compare_classifiers(X, y) -> pd.DataFrame:
    """
    Train multiple classifiers and compare results.
    Great for your PPT comparison table.
    """
    results = []
    for m in ["logistic", "random_forest", "xgboost"]:
        try:
            _, metrics = train_topic_classifier(X, y, model_type=m)
            results.append({
                "Model":        m,
                "Accuracy":     metrics["accuracy"],
                "F1 (macro)":   metrics["f1_macro"],
                "F1 (weighted)":metrics["f1_weighted"],
            })
        except Exception as e:
            logger.error(f"Failed {m}: {e}")

    df = pd.DataFrame(results)
    print("\n── Classifier Comparison ──────────────────────")
    print(df.to_string(index=False))
    return df


# ── Topic Importance Prediction ───────────────────────────────────────────────

def build_topic_importance_dataset(question_papers: list[dict]) -> pd.DataFrame:
    """
    Build a dataset for topic importance prediction from question papers.

    question_papers format:
      [
        { 'topic': str, 'frequency': int, 'total_marks': int,
          'recency_score': float, 'appeared_in_internal': bool,
          'appeared_in_endsem': bool, 'is_important': int (0/1) }
      ]

    is_important label should be manually annotated for training.
    """
    return pd.DataFrame(question_papers)


def train_importance_predictor(df: pd.DataFrame):
    """
    Train XGBoost to predict whether a topic is important.

    Features: frequency, total_marks, recency_score, appeared_in_*
    Label: is_important (0 or 1)
    """
    feature_cols = ['frequency', 'total_marks', 'recency_score',
                    'appeared_in_internal', 'appeared_in_endsem']

    # Keep only available features
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df['is_important'].values

    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "features":  available,
    }

    print("\n── Topic Importance Predictor (XGBoost) ────────")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['f1']}")

    return model, metrics


# ── Unsupervised Clustering ───────────────────────────────────────────────────

def cluster_topics(embeddings: np.ndarray, texts: list[str],
                   n_clusters: int = 5, method: str = "kmeans") -> pd.DataFrame:
    """
    Cluster note chunks or questions by semantic similarity.

    Args:
        embeddings: (n, dim) array
        texts: corresponding text list
        n_clusters: number of clusters
        method: "kmeans" | "hierarchical" | "dbscan"

    Returns:
        DataFrame with text and cluster label
    """
    from sklearn.metrics import silhouette_score

    labels = _cluster(embeddings, n_clusters, method)

    df = pd.DataFrame({'text': texts, 'cluster': labels})

    # Silhouette score (not valid for DBSCAN with noise)
    try:
        sil = silhouette_score(embeddings, labels)
        print(f"\n── Clustering ({method}, k={n_clusters}) ────────────")
        print(f"  Silhouette Score: {sil:.4f}  (higher is better, max=1)")
        df.attrs["silhouette"] = round(sil, 4)
    except Exception:
        pass

    return df


def _cluster(X: np.ndarray, n_clusters: int, method: str):
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return km.fit_predict(X)
    elif method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        return hc.fit_predict(X)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
        return db.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def find_optimal_k(embeddings: np.ndarray, k_range: range = range(2, 10)) -> dict:
    """
    Find best number of clusters using elbow method + silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(embeddings, labels)
            silhouettes.append(sil)
        except Exception:
            silhouettes.append(0)

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"\nBest k by silhouette score: {best_k}")

    return {
        "k_values":     list(k_range),
        "inertias":     inertias,
        "silhouettes":  silhouettes,
        "best_k":       best_k,
    }


# ── Model Persistence ────────────────────────────────────────────────────────

def save_model(model, path: str):
    """Save any sklearn/XGBoost model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {path}")


def load_model(path: str):
    """Load a saved model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
