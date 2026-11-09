"""
07_gerar_submissao.py
Arvore de Decisao otimizada por F1-macro
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessamento_utils import carregar_e_preparar

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "kaggle-training set", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "kaggle-test", "test.csv")
OUT_DIR    = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("GERANDO ARQUIVO DE SUBMISSAO PARA O KAGGLE")
    print("ESTRATEGIA: ARVORE DE DECISAO OTIMIZADA POR F1-MACRO")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, _, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    X_treino_completo = np.vstack([X_tr, X_val])
    y_treino_completo = np.concatenate([y_tr, y_val])

    candidatos = [
        {"max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": None, "min_samples_split": 5,  "min_samples_leaf": 2},
        {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 4},
        {"max_depth": 3,    "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 7,    "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 10,   "min_samples_split": 2,  "min_samples_leaf": 1},
        {"max_depth": 5,    "min_samples_split": 5,  "min_samples_leaf": 2},
        {"max_depth": 7,    "min_samples_split": 5,  "min_samples_leaf": 2},
        {"max_depth": 10,   "min_samples_split": 5,  "min_samples_leaf": 2},
        {"max_depth": 5,    "min_samples_split": 10, "min_samples_leaf": 4},
        {"max_depth": 7,    "min_samples_split": 10, "min_samples_leaf": 4},
        {"max_depth": 10,   "min_samples_split": 10, "min_samples_leaf": 4},
    ]

    print(f"\n[1] Buscando melhor configuracao ({len(candidatos)} candidatos, 5-fold CV, F1-macro)...\n")

    melhor_config = None
    melhor_f1 = -1

    for cfg in candidatos:
        dt = DecisionTreeClassifier(
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=42
        )
        scores = cross_val_score(dt, X_treino_completo, y_treino_completo,
                                 cv=5, scoring="f1_macro")
        media = scores.mean()
        print(f"   depth={str(cfg['max_depth']):4s} split={cfg['min_samples_split']:2d} leaf={cfg['min_samples_leaf']:2d} -> F1-macro={media:.4f}")
        if media > melhor_f1:
            melhor_f1 = media
            melhor_config = cfg

    print(f"\n   >>> Melhor configuracao:")
    print(f"       max_depth        = {melhor_config['max_depth']}")
    print(f"       min_samples_split = {melhor_config['min_samples_split']}")
    print(f"       min_samples_leaf  = {melhor_config['min_samples_leaf']}")
    print(f"       F1-macro CV       = {melhor_f1:.4f}")

    print("\n[2] Treinando modelo final...")
    dt_final = DecisionTreeClassifier(
        max_depth=melhor_config["max_depth"],
        min_samples_split=melhor_config["min_samples_split"],
        min_samples_leaf=melhor_config["min_samples_leaf"],
        random_state=42
    )
    dt_final.fit(X_treino_completo, y_treino_completo)
    print("   Treinamento concluido.")

    print("\n[3] Gerando predicoes...")
    predicoes = dt_final.predict(X_test)
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

    print(f"\n[4] Arquivo gerado: resultados/submission.csv")
    print(f"    Total de predicoes : {len(submission)}")
    print(f"    Controle (0)       : {(submission['class'] == 0).sum()}")
    print(f"    Diabetes (1)       : {(submission['class'] == 1).sum()}")
    print(f"\nEnvie o arquivo resultados/submission.csv para o Kaggle.")


if __name__ == "__main__":
    main()
