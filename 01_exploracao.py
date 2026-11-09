"""
01_exploracao.py
Analise exploratoria da base de dados de deteccao de diabetes.
"""

import pandas as pd
import numpy as np
import os

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "kaggle-training set", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "kaggle-test", "test.csv")

# ── Carregar dados ─────────────────────────────────────────────────────────────
print("=" * 60)
print("ANALISE EXPLORATORIA - DETECCAO DE DIABETES")
print("=" * 60)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"\nTamanho do treino : {train.shape[0]} linhas x {train.shape[1]} colunas")
print(f"Tamanho do teste  : {test.shape[0]} linhas x {test.shape[1]} colunas")

# ── Primeiras linhas ───────────────────────────────────────────────────────────
print("\n--- Primeiras linhas do treino ---")
print(train.head())

# ── Tipos de dados ─────────────────────────────────────────────────────────────
print("\n--- Tipos de dados ---")
print(train.dtypes)

# ── Valores ausentes ───────────────────────────────────────────────────────────
print("\n--- Valores ausentes por coluna ---")
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
df_missing = pd.DataFrame({"Ausentes": missing, "Percentual (%)": missing_pct})
df_missing = df_missing[df_missing["Ausentes"] > 0]
if df_missing.empty:
    print("Nenhum valor ausente encontrado.")
else:
    print(df_missing.to_string())

# ── Distribuicao da variavel-alvo ──────────────────────────────────────────────
print("\n--- Distribuicao da variavel-alvo (class) ---")
contagem = train["class"].value_counts()
percentual = train["class"].value_counts(normalize=True) * 100
df_alvo = pd.DataFrame({"Contagem": contagem, "Percentual (%)": percentual.round(2)})
df_alvo.index = df_alvo.index.map({0: "Controle (0)", 1: "Diabetes (1)"})
print(df_alvo.to_string())

# ── Estatisticas descritivas das colunas numericas ────────────────────────────
numericas = train.select_dtypes(include=[np.number]).columns.tolist()
numericas = [c for c in numericas if c not in ["id", "BIRTH_YEAR", "class"]]

print("\n--- Estatisticas descritivas (colunas numericas) ---")
print(train[numericas].describe().round(2).to_string())

# ── Salvar resumo em arquivo ───────────────────────────────────────────────────
OUT = os.path.join(BASE_DIR, "resultados", "01_exploracao.txt")
with open(OUT, "w") as f:
    f.write("ANALISE EXPLORATORIA - DETECCAO DE DIABETES\n")
    f.write(f"Treino: {train.shape}\n")
    f.write(f"Teste : {test.shape}\n\n")
    f.write("Distribuicao da variavel-alvo:\n")
    f.write(df_alvo.to_string())
    f.write("\n\nEstatisticas descritivas:\n")
    f.write(train[numericas].describe().round(2).to_string())

print(f"\nResumo salvo em: resultados/01_exploracao.txt")
