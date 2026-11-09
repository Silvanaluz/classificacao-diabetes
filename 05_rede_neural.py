"""
07_gerar_submissao.py
MLP otimizado por F1-macro
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
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
    print("ESTRATEGIA: MLP OTIMIZADO POR F1-MACRO")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, _, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    X_treino_completo = np.vstack([X_tr, X_val])
    y_treino_completo = np.concatenate([y_tr, y_val])

    candidatos = [
        {"hidden_layer_sizes": (64, 32),      "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32),      "alpha": 0.001,  "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32),      "alpha": 0.01,   "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64),     "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64),     "alpha": 0.001,  "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64),     "alpha": 0.01,   "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.001,  "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32),      "alpha": 0.001,  "learning_rate_init": 0.0005},
        {"hidden_layer_sizes": (128, 64),     "alpha": 0.001,  "learning_rate_init": 0.0005},
        {"hidden_layer_sizes": (64,),         "alpha": 0.001,  "learning_rate_init": 0.001},
    ]

    print(f"\n[1] Buscando melhor configuracao ({len(candidatos)} candidatos, 5-fold CV, F1-macro)...\n")

    melhor_config = None
    melhor_f1 = -1

    for cfg in candidatos:
        mlp = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        scores = cross_val_score(mlp, X_treino_completo, y_treino_completo,
                                 cv=5, scoring="f1_macro")
        media = scores.mean()
        print(f"   camadas={str(cfg['hidden_layer_sizes']):15s} alpha={cfg['alpha']:.4f} lr={cfg['learning_rate_init']:.4f} -> F1-macro={media:.4f}")
        if media > melhor_f1:
            melhor_f1 = media
            melhor_config = cfg

    print(f"\n   >>> Melhor configuracao:")
    print(f"       hidden_layer_sizes = {melhor_config['hidden_layer_sizes']}")
    print(f"       alpha              = {melhor_config['alpha']}")
    print(f"       learning_rate_init = {melhor_config['learning_rate_init']}")
    print(f"       F1-macro CV        = {melhor_f1:.4f}")

    print("\n[2] Treinando modelo final...")
    mlp_final = MLPClassifier(
        hidden_layer_sizes=melhor_config["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=melhor_config["alpha"],
        learning_rate_init=melhor_config["learning_rate_init"],
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    mlp_final.fit(X_treino_completo, y_treino_completo)
    print("   Treinamento concluido.")

    print("\n[3] Gerando predicoes...")
    predicoes = mlp_final.predict(X_test)
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
