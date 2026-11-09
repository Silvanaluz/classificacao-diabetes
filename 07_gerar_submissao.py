"""
07_gerar_submissao.py
Ensemble KNN + Arvore + MLP - media de probabilidades - F1-macro
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessamento_utils import carregar_e_preparar

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "kaggle-training set", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "kaggle-test", "test.csv")
OUT_DIR    = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)


def melhor_knn(X, y):
    print("   [KNN] buscando melhor K...")
    melhor_k, melhor_f1 = 1, -1
    for k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]:
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                                 X, y, cv=5, scoring="f1_macro")
        if scores.mean() > melhor_f1:
            melhor_f1 = scores.mean()
            melhor_k = k
    print(f"   [KNN] melhor K={melhor_k}  F1-macro={melhor_f1:.4f}")
    modelo = KNeighborsClassifier(n_neighbors=melhor_k)
    modelo.fit(X, y)
    return modelo


def melhor_arvore(X, y):
    print("   [Arvore] buscando melhor configuracao...")
    candidatos = [
        {"max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 3,    "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 7,    "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 5,    "min_samples_split": 5,  "min_samples_leaf": 2},
        {"max_depth": 7,    "min_samples_split": 5,  "min_samples_leaf": 2},
        {"max_depth": 10,   "min_samples_split": 5,  "min_samples_leaf": 2},
    ]
    melhor_cfg, melhor_f1 = None, -1
    from sklearn.tree import DecisionTreeClassifier
    for cfg in candidatos:
        scores = cross_val_score(
            DecisionTreeClassifier(random_state=42, **cfg),
            X, y, cv=5, scoring="f1_macro")
        if scores.mean() > melhor_f1:
            melhor_f1 = scores.mean()
            melhor_cfg = cfg
    print(f"   [Arvore] melhor depth={melhor_cfg['max_depth']}  F1-macro={melhor_f1:.4f}")
    modelo = DecisionTreeClassifier(random_state=42, **melhor_cfg)
    modelo.fit(X, y)
    return modelo


def melhor_mlp(X, y):
    print("   [MLP] buscando melhor configuracao...")
    candidatos = [
        {"hidden_layer_sizes": (64, 32),  "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32),  "alpha": 0.001,  "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.001,  "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64,),     "alpha": 0.001,  "learning_rate_init": 0.001},
    ]
    melhor_cfg, melhor_f1 = None, -1
    from sklearn.neural_network import MLPClassifier
    for cfg in candidatos:
        mlp = MLPClassifier(activation="relu", solver="adam", max_iter=1000,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=20, random_state=42, **cfg)
        scores = cross_val_score(mlp, X, y, cv=5, scoring="f1_macro")
        if scores.mean() > melhor_f1:
            melhor_f1 = scores.mean()
            melhor_cfg = cfg
    print(f"   [MLP] melhor camadas={melhor_cfg['hidden_layer_sizes']}  F1-macro={melhor_f1:.4f}")
    modelo = MLPClassifier(activation="relu", solver="adam", max_iter=1000,
                           early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=20, random_state=42, **melhor_cfg)
    modelo.fit(X, y)
    return modelo


def main():
    print("=" * 60)
    print("GERANDO ARQUIVO DE SUBMISSAO PARA O KAGGLE")
    print("ESTRATEGIA: ENSEMBLE KNN + ARVORE + MLP (MEDIA PROB)")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, _, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    X_treino = np.vstack([X_tr, X_val])
    y_treino = np.concatenate([y_tr, y_val])

    print("\n[1] Treinando modelos otimizados por F1-macro...")
    knn  = melhor_knn(X_treino, y_treino)
    arv  = melhor_arvore(X_treino, y_treino)
    mlp  = melhor_mlp(X_treino, y_treino)

    print("\n[2] Gerando probabilidades e ensemble...")
    prob_knn = knn.predict_proba(X_test)
    prob_arv = arv.predict_proba(X_test)
    prob_mlp = mlp.predict_proba(X_test)

    prob_media = (prob_knn + prob_arv + prob_mlp) / 3.0
    predicoes  = np.argmax(prob_media, axis=1)

    print(f"   Controle (0): {(predicoes == 0).sum()} pacientes")
    print(f"   Diabetes (1): {(predicoes == 1).sum()} pacientes")

    test_raw = pd.read_csv(TEST_PATH)

    if os.path.exists(os.path.join(BASE_DIR, "submission_template.csv")):
        template = pd.read_csv(os.path.join(BASE_DIR, "submission_template.csv"))
        id_para_pred = dict(zip(test_raw["id"].astype(int), predicoes))
        template["class"] = template["id"].astype(int).map(id_para_pred)
        if template["class"].isna().sum() > 0:
            template["class"] = template["class"].fillna(0).astype(int)
        else:
            template["class"] = template["class"].astype(int)
        submission = template
    else:
        submission = pd.DataFrame({
            "id": test_raw["id"].astype(int),
            "class": predicoes
        })

    out_path = os.path.join(OUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)

    print(f"\n[3] Arquivo gerado: resultados/submission.csv")
    print(f"    Total de predicoes : {len(submission)}")
    print(f"    Controle (0)       : {(submission['class'] == 0).sum()}")
    print(f"    Diabetes (1)       : {(submission['class'] == 1).sum()}")
    print(f"\nEnvie o arquivo resultados/submission.csv para o Kaggle.")


if __name__ == "__main__":
    main()