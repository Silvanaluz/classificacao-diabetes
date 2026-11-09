"""
04_arvore_decisao.py
Modelo de Arvore de Decisao (Decision Tree) para deteccao de diabetes.
Busca profundidade ideal por validacao cruzada e apresenta importancia de features.
"""

import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
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
    print("MODELO ARVORE DE DECISAO")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, y_test, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    # ── Busca da melhor profundidade maxima ────────────────────────────────────
    print("\n[1] Buscando melhor max_depth (1 a 15)...")
    resultados_depth = []
    for d in range(1, 16):
        dt = DecisionTreeClassifier(
            max_depth=d,
            criterion="gini",
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        scores = cross_val_score(dt, X_tr, y_tr, cv=5, scoring="f1")
        resultados_depth.append((d, scores.mean(), scores.std()))
        print(f"   depth={d:2d} | F1 medio={scores.mean():.4f} (+/- {scores.std():.4f})")

    melhor_depth = max(resultados_depth, key=lambda x: x[1])[0]
    print(f"\n   Melhor profundidade encontrada: {melhor_depth}")

    # ── Treinar modelo final ───────────────────────────────────────────────────
    print("\n[2] Treinando modelo final com max_depth =", melhor_depth)
    dt = DecisionTreeClassifier(
        max_depth=melhor_depth,
        criterion="gini",
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    dt.fit(X_tr, y_tr)

    # ── Importancia das features ───────────────────────────────────────────────
    print("\n[3] Importancia das features (top 10):")
    importancias = sorted(
        zip(features, dt.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importancias[:10]:
        bar = "#" * int(imp * 50)
        print(f"   {feat:25s} {imp:.4f}  {bar}")

    # ── Avaliacoes ─────────────────────────────────────────────────────────────
    print("\n[4] Resultados:")
    r_tr  = avaliar(y_tr,  dt.predict(X_tr),  "Treino")
    r_val = avaliar(y_val, dt.predict(X_val), "Validacao")

    if y_test is not None:
        r_test = avaliar(y_test, dt.predict(X_test), "Teste Final")
        print("\n  Relatorio completo (Teste Final):")
        print(classification_report(y_test, dt.predict(X_test),
                                    target_names=["Controle (0)", "Diabetes (1)"]))

    # ── Salvar resultados ──────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "04_arvore_resultados.txt")
    with open(out_path, "w") as f:
        f.write(f"Melhor profundidade: {melhor_depth}\n\n")
        f.write("Importancia das features:\n")
        for feat, imp in importancias:
            f.write(f"  {feat}: {imp:.4f}\n")
        f.write("\n")
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

    print(f"\nResultados salvos em: resultados/04_arvore_resultados.txt")


if __name__ == "__main__":
    main()
