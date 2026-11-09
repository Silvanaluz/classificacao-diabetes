"""
05_rede_neural.py
Modelo de Rede Neural (MLP - Perceptron Multicamadas) para deteccao de diabetes.
Busca arquitetura e taxa de aprendizado por validacao cruzada.
"""

import sys
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessamento_utils import carregar_e_preparar

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "kaggle-training set", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "kaggle-test", "test.csv")
OUT_DIR    = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)


def avaliar(y_real, y_pred, nome_conjunto):
    acc = accuracy_score(y_real, y_pred)
    pre = precision_score(y_real, y_pred, zero_division=0)
    rec = recall_score(y_real, y_pred, zero_division=0)
    f1  = f1_score(y_real, y_pred, zero_division=0)
    cm  = confusion_matrix(y_real, y_pred)
    print(f"\n  [{nome_conjunto}]")
    print(f"  Acuracia : {acc:.4f}")
    print(f"  Precisao : {pre:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Matriz de Confusao:\n{cm}")
    return {"conjunto": nome_conjunto, "acuracia": acc, "precisao": pre,
            "recall": rec, "f1": f1}


def main():
    print("=" * 60)
    print("MODELO REDE NEURAL (MLP)")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, y_test, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    # ── Busca de arquitetura por validacao cruzada ─────────────────────────────
    print("\n[1] Buscando melhor arquitetura (5-fold CV)...")

    candidatos = [
        {"hidden_layer_sizes": (64,),       "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32),    "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64),   "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32, 16),"alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64,),       "alpha": 0.01,  "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32),    "alpha": 0.01,  "learning_rate_init": 0.001},
    ]

    melhor_config = None
    melhor_f1 = -1

    for cfg in candidatos:
        mlp = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42
        )
        scores = cross_val_score(mlp, X_tr, y_tr, cv=5, scoring="f1")
        media = scores.mean()
        print(f"   camadas={str(cfg['hidden_layer_sizes']):15s} alpha={cfg['alpha']}  "
              f"F1 medio={media:.4f} (+/- {scores.std():.4f})")
        if media > melhor_f1:
            melhor_f1 = media
            melhor_config = cfg

    print(f"\n   Melhor configuracao: camadas={melhor_config['hidden_layer_sizes']}, "
          f"alpha={melhor_config['alpha']}")

    # ── Treinar modelo final ───────────────────────────────────────────────────
    print("\n[2] Treinando modelo final...")
    mlp = MLPClassifier(
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
    mlp.fit(X_tr, y_tr)
    print(f"   Treinamento concluido em {mlp.n_iter_} iteracoes.")

    # ── Avaliacoes ─────────────────────────────────────────────────────────────
    print("\n[3] Resultados:")
    r_tr  = avaliar(y_tr,  mlp.predict(X_tr),  "Treino")
    r_val = avaliar(y_val, mlp.predict(X_val), "Validacao")

    if y_test is not None:
        r_test = avaliar(y_test, mlp.predict(X_test), "Teste Final")
        print("\n  Relatorio completo (Teste Final):")
        print(classification_report(y_test, mlp.predict(X_test),
                                    target_names=["Controle (0)", "Diabetes (1)"]))

    # ── Salvar resultados ──────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "05_rede_neural_resultados.txt")
    with open(out_path, "w") as f:
        f.write(f"Melhor configuracao: {melhor_config}\n")
        f.write(f"Iteracoes de treino: {mlp.n_iter_}\n\n")
        for r in [r_tr, r_val]:
            f.write(
                f"{r['conjunto']} | Acuracia={r['acuracia']:.4f} | "
                f"Precisao={r['precisao']:.4f} | Recall={r['recall']:.4f} | "
                f"F1={r['f1']:.4f}\n"
            )
        if y_test is not None:
            f.write(
                f"{r_test['conjunto']} | Acuracia={r_test['acuracia']:.4f} | "
                f"Precisao={r_test['precisao']:.4f} | Recall={r_test['recall']:.4f} | "
                f"F1={r_test['f1']:.4f}\n"
            )

    print(f"\nResultados salvos em: resultados/05_rede_neural_resultados.txt")


if __name__ == "__main__":
    main()
