"""
07_gerar_submissao.py
Submissao 8 - Gradient Boosting otimizado por F1-macro
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessamento_utils import carregar_e_preparar

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH    = os.path.join(BASE_DIR, "kaggle-training set", "train.csv")
TEST_PATH     = os.path.join(BASE_DIR, "kaggle-test", "test.csv")
OUT_DIR       = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("GERANDO ARQUIVO DE SUBMISSAO PARA O KAGGLE")
    print("ESTRATEGIA: GRADIENT BOOSTING OTIMIZADO POR F1-MACRO")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, _, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    X_treino_completo = np.vstack([X_tr, X_val])
    y_treino_completo = np.concatenate([y_tr, y_val])

    candidatos = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    ]

    print(f"\n[1] Buscando melhor configuracao ({len(candidatos)} candidatos, 5-fold CV, F1-macro)...\n")

    melhor_config = None
    melhor_f1 = -1

    for cfg in candidatos:
        gb = GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            random_state=42
        )
        scores = cross_val_score(gb, X_treino_completo, y_treino_completo,
                                 cv=5, scoring="f1_macro")
        media = scores.mean()
        print(f"   n={cfg['n_estimators']:3d} depth={cfg['max_depth']} "
              f"lr={cfg['learning_rate']:.2f} sub={cfg['subsample']:.1f} "
              f"-> F1-macro={media:.4f} (+/-{scores.std():.4f})")
        if media > melhor_f1:
            melhor_f1 = media
            melhor_config = cfg

    print(f"\n   >>> Melhor configuracao:")
    print(f"       n_estimators  = {melhor_config['n_estimators']}")
    print(f"       max_depth     = {melhor_config['max_depth']}")
    print(f"       learning_rate = {melhor_config['learning_rate']}")
    print(f"       subsample     = {melhor_config['subsample']}")
    print(f"       F1-macro CV   = {melhor_f1:.4f}")

    print("\n[2] Treinando modelo final...")
    gb_final = GradientBoostingClassifier(
        n_estimators=melhor_config["n_estimators"],
        max_depth=melhor_config["max_depth"],
        learning_rate=melhor_config["learning_rate"],
        subsample=melhor_config["subsample"],
        random_state=42
    )
    gb_final.fit(X_treino_completo, y_treino_completo)
    print("   Treinamento concluido.")

    print("\n[3] Gerando predicoes...")
    predicoes = gb_final.predict(X_test)
    print(f"   Controle (0): {(predicoes == 0).sum()} pacientes")
    print(f"   Diabetes (1): {(predicoes == 1).sum()} pacientes")

    test_raw = pd.read_csv(TEST_PATH)

    if os.path.exists(os.path.join(BASE_DIR, "submission_template.csv")):
        template = pd.read_csv(os.path.join(BASE_DIR, "submission_template.csv"))
        id_para_pred = dict(zip(test_raw["id"].astype(int), predicoes))
        template["class"] = template["id"].astype(int).map(id_para_pred)
        sem_pred = template["class"].isna().sum()
        if sem_pred > 0:
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