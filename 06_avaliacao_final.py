"""
06_avaliacao_final.py
Avaliacao final comparando os tres modelos no conjunto de teste.
Gera tabela comparativa e resumo para apresentacao.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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


def metricas(y_real, y_pred):
    return {
        "Acuracia" : round(accuracy_score(y_real, y_pred), 4),
        "Precisao" : round(precision_score(y_real, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_real, y_pred, zero_division=0), 4),
        "F1-Score" : round(f1_score(y_real, y_pred, zero_division=0), 4),
    }


def main():
    print("=" * 60)
    print("AVALIACAO FINAL - COMPARACAO DOS TRES MODELOS")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val, X_test, y_test, features = carregar_e_preparar(
        TRAIN_PATH, TEST_PATH
    )

    # Combinar treino + validacao para treinar o modelo final
    X_treino_completo = np.vstack([X_tr, X_val])
    y_treino_completo = np.concatenate([y_tr, y_val])

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Arvore de Decisao": DecisionTreeClassifier(
            max_depth=5, criterion="gini",
            min_samples_split=5, min_samples_leaf=3, random_state=42
        ),
        "Rede Neural (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001,
            max_iter=1000, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
            random_state=42
        ),
    }

    # Nota: os parametros acima sao defaults razoaveis.
    # Para resultados otimos, substitua pelos melhores parametros
    # encontrados pelos scripts 03, 04 e 05.

    print("\nTreinando e avaliando os tres modelos...\n")
    resultados = []

    for nome, modelo in modelos.items():
        print(f"[{nome}]")

        # Treinar no conjunto completo (treino + validacao)
        modelo.fit(X_treino_completo, y_treino_completo)

        # Avaliar no treino completo
        r_tr = metricas(y_treino_completo, modelo.predict(X_treino_completo))

        # Avaliar no teste (avaliacao final)
        if y_test is not None:
            r_te = metricas(y_test, modelo.predict(X_test))
            print(f"  Treino -> F1: {r_tr['F1-Score']}  |  "
                  f"Teste -> F1: {r_te['F1-Score']}")
            resultados.append({
                "Modelo": nome,
                "Acuracia_treino": r_tr["Acuracia"],
                "F1_treino": r_tr["F1-Score"],
                "Acuracia_teste": r_te["Acuracia"],
                "Precisao_teste": r_te["Precisao"],
                "Recall_teste": r_te["Recall"],
                "F1_teste": r_te["F1-Score"],
            })

            print(f"  Matriz de Confusao (Teste):")
            print(confusion_matrix(y_test, modelo.predict(X_test)))
            print()
        else:
            print(f"  Treino -> F1: {r_tr['F1-Score']}")
            print("  (conjunto de teste sem rotulo - avaliacao externa sera feita separadamente)")
            resultados.append({
                "Modelo": nome,
                "Acuracia_treino": r_tr["Acuracia"],
                "F1_treino": r_tr["F1-Score"],
            })

    # ── Tabela comparativa ─────────────────────────────────────────────────────
    df_res = pd.DataFrame(resultados)
    print("\n" + "=" * 60)
    print("TABELA COMPARATIVA FINAL")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # ── Salvar ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "06_comparacao_modelos.txt")
    with open(out_path, "w") as f:
        f.write("TABELA COMPARATIVA - DETECCAO DE DIABETES\n")
        f.write("=" * 60 + "\n")
        f.write(df_res.to_string(index=False))
        f.write("\n")

    out_csv = os.path.join(OUT_DIR, "06_comparacao_modelos.csv")
    df_res.to_csv(out_csv, index=False)

    print(f"\nResultados salvos em: resultados/06_comparacao_modelos.txt")
    print(f"CSV salvo em        : resultados/06_comparacao_modelos.csv")

    # ── Indicar melhor modelo (por F1 no teste) ────────────────────────────────
    if "F1_teste" in df_res.columns:
        melhor = df_res.loc[df_res["F1_teste"].idxmax(), "Modelo"]
        print(f"\nMelhor modelo (F1 no teste): {melhor}")


if __name__ == "__main__":
    main()
