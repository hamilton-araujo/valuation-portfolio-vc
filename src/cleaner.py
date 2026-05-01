"""
Limpeza e pré-processamento do dataset de investimentos.

Pipeline:
    1. Downcasting de tipos numéricos (economia de RAM)
    2. Filtro de estágios válidos (Seed → Series C)
    3. Remoção de outliers via IQR na escala logarítmica
    4. Normalização de nomes de colunas e estágios
    5. Engenharia de features básicas (ano, dias até funding)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ESTAGIOS_MAP = {
    "seed":           "Seed",
    "angel":          "Seed",
    "venture":        "Seed",
    "series-a":       "Series A",
    "series-b":       "Series B",
    "series-c":       "Series C",
    "series-c+":      "Series C",
}

ESTAGIOS_ORDEM = ["Seed", "Series A", "Series B", "Series C"]

# Probabilidades de falha calibradas por estágio (literatura VC)
PROB_FALHA = {
    "Seed":     0.70,
    "Series A": 0.50,
    "Series B": 0.35,
    "Series C": 0.25,
}


def limpar(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Executa o pipeline completo de limpeza.

    Args:
        df:               DataFrame bruto da ingestão.
        iqr_multiplier:   Multiplicador do IQR para detecção de outliers.

    Returns:
        DataFrame limpo e pronto para imputação/simulação.
    """
    logger.info("Iniciando limpeza. Shape inicial: %s", df.shape)

    df = _normalizar_colunas(df)
    df = _filtrar_estagios(df)
    df = _downcast(df)
    df = _remover_outliers_iqr(df, "raised_amount_usd", iqr_multiplier)
    df = _feature_engineering(df)
    df = df.reset_index(drop=True)

    logger.info("Limpeza concluída. Shape final: %s", df.shape)
    return df


# ─────────────────────────────────────────────
# Etapas internas
# ─────────────────────────────────────────────

def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas e mapeia estágios para categorias."""
    df = df.copy()

    # Normalizar nome de coluna de estágio
    if "funding_round_type" in df.columns:
        df["estagio"] = (
            df["funding_round_type"]
            .str.lower()
            .str.strip()
            .map(ESTAGIOS_MAP)
        )
    elif "funding_round_permalink" in df.columns:
        df["estagio"] = df["funding_round_permalink"].map(ESTAGIOS_MAP)

    # País
    if "company_country_code" in df.columns:
        df["pais"] = df["company_country_code"].str.upper().str.strip()
    elif "country_code" in df.columns:
        df["pais"] = df["country_code"].str.upper().str.strip()

    # Setor
    if "company_market" in df.columns:
        df["setor"] = df["company_market"].str.strip().str.title()
    elif "market" in df.columns:
        df["setor"] = df["market"].str.strip().str.title()

    # Status da empresa
    if "status" in df.columns:
        df["saiu"] = df["status"].isin(["acquired", "ipo"]).astype("int8")

    return df


def _filtrar_estagios(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas rodadas dos estágios definidos em ESTAGIOS_ORDEM."""
    antes = len(df)
    df = df[df["estagio"].isin(ESTAGIOS_ORDEM)].copy()
    df["estagio"] = pd.Categorical(df["estagio"], categories=ESTAGIOS_ORDEM, ordered=True)
    logger.info("Filtro de estágio: %d → %d linhas", antes, len(df))
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz uso de memória convertendo tipos numéricos para o menor tipo válido.
    Float64 → Float32, Int64 → Int32/Int16 onde possível.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    uso_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info("Memória após downcast: %.1f MB", uso_mb)
    return df


def _remover_outliers_iqr(
    df: pd.DataFrame,
    coluna: str,
    multiplicador: float,
) -> pd.DataFrame:
    """
    Remove outliers severos via IQR aplicado na escala logarítmica.

    Usar escala log é essencial para distribuições de captação de capital,
    que são fortemente assimétricas (Power Law). O IQR linear eliminaria
    empresas legítimas com captações altas.
    """
    df = df.copy()
    serie = df[coluna].dropna()
    serie = serie[serie > 0]

    log_serie = np.log(serie)
    q1, q3 = log_serie.quantile(0.25), log_serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf = np.exp(q1 - multiplicador * iqr)
    limite_sup = np.exp(q3 + multiplicador * iqr)

    antes = len(df)
    mascara = (
        df[coluna].isna() |
        ((df[coluna] >= limite_inf) & (df[coluna] <= limite_sup))
    )
    df = df[mascara].copy()

    logger.info(
        "IQR log (×%.1f): removidos %d outliers em '%s'. "
        "Limites: [$%.0f, $%.0f]",
        multiplicador, antes - len(df), coluna, limite_inf, limite_sup,
    )
    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas usadas na imputação e simulação."""
    df = df.copy()

    # Ano do funding
    if "funded_at" in df.columns:
        df["ano_funding"] = pd.to_datetime(df["funded_at"], errors="coerce").dt.year
        df["ano_funding"] = pd.to_numeric(df["ano_funding"], downcast="integer")

    # Log do capital levantado (feature para Ridge)
    if "raised_amount_usd" in df.columns:
        df["log_raised"] = np.log1p(df["raised_amount_usd"].clip(lower=0))

    # Estágio como ordinal numérico
    df["estagio_num"] = df["estagio"].cat.codes.astype("int8")

    # Probabilidade de falha por estágio
    df["prob_falha"] = df["estagio"].map(PROB_FALHA).astype("float32")

    return df
