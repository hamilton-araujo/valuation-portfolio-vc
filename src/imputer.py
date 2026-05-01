"""
Imputação de valuation pós-money via Ridge Regression segmentada por setor.

Pipeline:
    1. Separar registros com/sem valuation conhecido
    2. Construir features: log_raised, estagio_num, pais, ano_funding
    3. Treinar Ridge (α=L2) por setor via K-Fold Cross-Validation
    4. Imputar valores faltantes e reinserir no DataFrame
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

ALPHA = 1.0
N_SPLITS = 5
MIN_AMOSTRAS = 30  # Mínimo para treinar modelo por setor

FEATURES_NUM = ["log_raised", "estagio_num", "ano_funding"]
FEATURES_CAT = ["pais"]


def imputar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valuations faltantes usando Ridge Regression segmentada por setor.

    Args:
        df: DataFrame limpo (saída de cleaner.limpar).

    Returns:
        DataFrame com coluna 'valuation_imputado' preenchida.
    """
    df = df.copy()

    if "raised_amount_usd" not in df.columns:
        logger.warning("Coluna 'raised_amount_usd' ausente — imputação abortada.")
        return df

    # Usar raised_amount_usd como proxy de valuation quando não há coluna dedicada
    if "valuation" not in df.columns:
        logger.info("Coluna 'valuation' não encontrada — usando raised_amount_usd como alvo.")
        df["valuation"] = np.nan

    df["log_valuation"] = np.where(
        df["valuation"] > 0,
        np.log(df["valuation"]),
        np.nan,
    )

    # Imputar globalmente como fallback
    modelo_global, metricas_global = _treinar_modelo(df)
    df = _imputar_com_modelo(df, modelo_global, sufixo="global")

    # Imputar por setor quando há dados suficientes
    if "setor" in df.columns:
        setores = df["setor"].dropna().unique()
        for setor in setores:
            mask_setor = df["setor"] == setor
            df_setor = df[mask_setor]
            n_com_val = df_setor["log_valuation"].notna().sum()

            if n_com_val < MIN_AMOSTRAS:
                logger.debug("Setor '%s': apenas %d amostras — usando modelo global.", setor, n_com_val)
                continue

            modelo_setor, metricas_setor = _treinar_modelo(df_setor)
            logger.info(
                "Setor '%s': R²=%.3f  MAE=%.3f  (n=%d)",
                setor,
                metricas_setor["r2"],
                metricas_setor["mae"],
                n_com_val,
            )

            # Sobrescrever imputação global apenas para startups deste setor sem valuation
            idx_faltantes = df.index[mask_setor & df["log_valuation"].isna()]
            if len(idx_faltantes) == 0:
                continue

            X_pred = _preparar_features(df.loc[idx_faltantes])
            pred_log = modelo_setor.predict(X_pred)
            df.loc[idx_faltantes, "valuation_imputado"] = np.exp(pred_log)

    logger.info(
        "Imputação concluída. Global — R²=%.3f  MAE=%.3f",
        metricas_global["r2"],
        metricas_global["mae"],
    )
    return df


def _treinar_modelo(df: pd.DataFrame):
    """Treina Ridge via K-Fold e retorna (pipeline_treinado, metricas)."""
    df_treino = df[df["log_valuation"].notna()].copy()

    if len(df_treino) < MIN_AMOSTRAS:
        return _modelo_fallback(), {"r2": np.nan, "mae": np.nan}

    X = _preparar_features(df_treino)
    y = df_treino["log_valuation"].values

    pipeline = _construir_pipeline()

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
        return_train_score=False,
    )

    metricas = {
        "r2":  float(np.mean(scores["test_r2"])),
        "mae": float(-np.mean(scores["test_mae"])),
    }

    pipeline.fit(X, y)
    return pipeline, metricas


def _imputar_com_modelo(df: pd.DataFrame, modelo, sufixo: str) -> pd.DataFrame:
    """Aplica modelo a todas as linhas sem valuation e salva em valuation_imputado."""
    mask_faltante = df["log_valuation"].isna()
    if not mask_faltante.any():
        df["valuation_imputado"] = df["valuation"]
        return df

    # Valores conhecidos passam direto
    df["valuation_imputado"] = df["valuation"]

    idx_faltantes = df.index[mask_faltante]
    X_pred = _preparar_features(df.loc[idx_faltantes])
    pred_log = modelo.predict(X_pred)
    df.loc[idx_faltantes, "valuation_imputado"] = np.exp(pred_log)

    logger.info(
        "Modelo %s: %d valuations imputados.",
        sufixo, mask_faltante.sum(),
    )
    return df


def _preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e devolve sub-DataFrame com features numéricas + categóricas."""
    cols = []
    for c in FEATURES_NUM + FEATURES_CAT:
        if c in df.columns:
            cols.append(c)
        else:
            df = df.copy()
            df[c] = np.nan
            cols.append(c)
    return df[cols]


def _construir_pipeline() -> Pipeline:
    """Constrói preprocessador + Ridge em um sklearn Pipeline."""
    num_cols_presentes = FEATURES_NUM
    cat_cols_presentes = FEATURES_CAT

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",  StandardScaler()),
                ]),
                num_cols_presentes,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols_presentes,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("ridge",        Ridge(alpha=ALPHA)),
    ])


def _modelo_fallback():
    """Modelo que retorna a mediana — usado quando dados insuficientes."""
    class MedianaFallback:
        def __init__(self):
            self._mediana = None

        def fit(self, X, y):
            self._mediana = float(np.nanmedian(y))
            return self

        def predict(self, X):
            return np.full(len(X), self._mediana if self._mediana else 0.0)

    return MedianaFallback()
