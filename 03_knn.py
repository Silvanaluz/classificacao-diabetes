"""
03_knn.py
Modelo K-Vizinhos Mais Proximos (KNN) para deteccao de diabetes.
Busca o melhor K por validacao cruzada no conjunto de treino.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Importar pre-processamento
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
    print("MODELO KNN - K-VIZINHOS MAIS PROXIMOS")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, y_test, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    # ── Busca do melhor K (validacao cruzada 5-fold no treino) ─────────────────
    print("\n[1] Buscando melhor K (K impar de 1 a 21)...")
    k_valores = range(1, 22, 2)  # 1, 3, 5, ..., 21
    resultados_k = []
    for k in k_valores:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        scores = cross_val_score(knn, X_tr, y_tr, cv=5, scoring="f1")
        resultados_k.append((k, scores.mean(), scores.std()))
        print(f"   K={k:2d} | F1 medio={scores.mean():.4f} (+/- {scores.std():.4f})")

    melhor_k = max(resultados_k, key=lambda x: x[1])[0]
    print(f"\n   Melhor K encontrado: {melhor_k}")

    # ── Treinar modelo final com o melhor K ────────────────────────────────────
    print("\n[2] Treinando modelo final com K =", melhor_k)
    knn = KNeighborsClassifier(n_neighbors=melhor_k, metric="euclidean")
    knn.fit(X_tr, y_tr)

    # ── Avaliacoes ─────────────────────────────────────────────────────────────
    print("\n[3] Resultados:")
    r_tr  = avaliar(y_tr,  knn.predict(X_tr),  "Treino")
    r_val = avaliar(y_val, knn.predict(X_val), "Validacao")

    if y_test is not None:
        r_test = avaliar(y_test, knn.predict(X_test), "Teste Final")
        print("\n  Relatorio completo (Teste Final):")
        print(classification_report(y_test, knn.predict(X_test),
                                    target_names=["Controle (0)", "Diabetes (1)"]))

    # ── Salvar resultados ──────────────────────────────────────────────────────
    linhas = [f"Melhor K: {melhor_k}\n"]
    for r in [r_tr, r_val]:
        linhas.append(
            f"{r['conjunto']} | Acuracia={r['acuracia']:.4f} | "
            f"Precisao={r['precisao']:.4f} | Recall={r['recall']:.4f} | "
            f"F1={r['f1']:.4f}\n"
        )
    if y_test is not None:
        linhas.append(
            f"{r_test['conjunto']} | Acuracia={r_test['acuracia']:.4f} | "
            f"Precisao={r_test['precisao']:.4f} | Recall={r_test['recall']:.4f} | "
            f"F1={r_test['f1']:.4f}\n"
        )

    with open(os.path.join(OUT_DIR, "03_knn_resultados.txt"), "w") as f:
        f.writelines(linhas)

    print("\nResultados salvos em: resultados/03_knn_resultados.txt")


if __name__ == "__main__":
    main()
