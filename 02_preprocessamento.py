"""
02_preprocessamento.py
Pre-processamento compartilhado por todos os modelos.
Exporta a funcao: carregar_e_preparar(train_path, test_path)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os


# Colunas que sao identificadores ou irrelevantes para o modelo
COLUNAS_REMOVER = ["id", "PATIENT_CODE", "EXAM_DATE", "BIRTH_YEAR", "COLLECTION_SITE", "LAB_DEVICE"]


# Coluna alvo
COLUNA_ALVO = "class"


def _limpar_pressao(valor):
    """Converte BLOOD_PRESSURE 'sistolica/diastolica' em valor sistolico (float)."""
    if pd.isna(valor):
        return np.nan
    s = str(valor)
    if "/" in s:
        try:
            return float(s.split("/")[0])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def carregar_e_preparar(train_path, test_path, val_size=0.2, random_state=42):
    """
    Carrega, limpa e prepara os dados.

    Retorna:
        X_train, X_val, y_train, y_val : treino e validacao (para ajuste de modelos)
        X_test, y_test                 : teste final (NAO usar ate avaliacao final)
        feature_names                  : lista de nomes de colunas apos pre-processamento
    """
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    # ── Separar alvo ──────────────────────────────────────────────────────────
    # O conjunto de teste pode ou nao conter a coluna 'class'
    y_test = None
    if COLUNA_ALVO in test_raw.columns:
        y_test = test_raw[COLUNA_ALVO].values

    # ── Remover colunas identificadoras ───────────────────────────────────────
    colunas_existentes_remover = [c for c in COLUNAS_REMOVER if c in train_raw.columns]
    train_raw = train_raw.drop(columns=colunas_existentes_remover, errors="ignore")
    test_raw  = test_raw.drop(columns=colunas_existentes_remover, errors="ignore")

    # ── Tratar BLOOD_PRESSURE ─────────────────────────────────────────────────
    if "BLOOD_PRESSURE" in train_raw.columns:
        train_raw["BLOOD_PRESSURE"] = train_raw["BLOOD_PRESSURE"].apply(_limpar_pressao)
        test_raw["BLOOD_PRESSURE"]  = test_raw["BLOOD_PRESSURE"].apply(_limpar_pressao)

    # ── Separar X e y ─────────────────────────────────────────────────────────
    y_train_full = train_raw[COLUNA_ALVO].values
    X_train_full = train_raw.drop(columns=[COLUNA_ALVO], errors="ignore")

    X_test_df = test_raw.drop(columns=[COLUNA_ALVO], errors="ignore")

    # ── Codificar colunas categoricas (Label Encoding) ────────────────────────
    categoricas = X_train_full.select_dtypes(include=["object"]).columns.tolist()

    encoders = {}
    for col in categoricas:
        le = LabelEncoder()
        all_values = pd.concat([X_train_full[col], X_test_df[col]], axis=0)
        all_values = all_values.fillna("MISSING").astype(str)
        le.fit(all_values)
        X_train_full[col] = le.transform(
            X_train_full[col].fillna("MISSING").astype(str)
        )
        X_test_df[col] = le.transform(
            X_test_df[col].fillna("MISSING").astype(str)
        )
        encoders[col] = le

    # ── Preencher valores ausentes numericos com mediana ──────────────────────
    numericas = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
    medianas = X_train_full[numericas].median()
    X_train_full[numericas] = X_train_full[numericas].fillna(medianas)
    X_test_df[numericas]    = X_test_df[numericas].fillna(medianas)

    # ── Dividir treino em treino + validacao ──────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full.values,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full
    )

    # ── Normalizar (StandardScaler ajustado APENAS no treino) ─────────────────
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test_df.values)

    feature_names = list(X_train_full.columns)

    print(f"[Preprocessamento] Treino: {X_tr.shape} | Validacao: {X_val.shape} | Teste: {X_test_norm.shape}")
    print(f"[Preprocessamento] Features utilizadas ({len(feature_names)}): {feature_names}")

    return X_tr, X_val, y_tr, y_val, X_test_norm, y_test, feature_names
