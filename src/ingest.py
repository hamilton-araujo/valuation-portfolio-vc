"""
Ingestão do Startup Investments Dataset via Kaggle API.

Download:
    kaggle datasets download -d justinas/startup-investments
    Descompactar em data/raw/

Cache:
    Salva em data/investments.parquet para evitar reprocessamento.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_DIR   = DATA_DIR / "raw"
CACHE     = DATA_DIR / "investments.parquet"

# Colunas que serão mantidas do dataset
COLUNAS_UTEIS = [
    "permalink", "name", "category_list", "market",
    "country_code", "state_code", "region", "city",
    "funding_total_usd", "status",
    "funding_rounds", "founded_at",
    "first_funding_at", "last_funding_at",
]

COLUNAS_ROUNDS = [
    "company_permalink", "company_name", "company_market",
    "company_country_code", "funding_round_type",
    "funded_at", "raised_amount_usd",
]

ESTAGIOS_VALIDOS = {"seed", "series-a", "series-b", "series-c", "angel", "venture"}


def carregar(force_reload: bool = False) -> pd.DataFrame:
    """
    Carrega o dataset processado do cache ou reconstrói a partir dos CSVs brutos.

    Returns:
        DataFrame com rodadas de financiamento enriquecidas com dados da empresa.
    """
    if not force_reload and CACHE.exists():
        logger.info("Carregando cache: %s", CACHE.name)
        return pd.read_parquet(CACHE)

    df = _construir_do_zero()
    df.to_parquet(CACHE, index=False)
    logger.info("Cache salvo: %s (%d linhas)", CACHE.name, len(df))
    return df


def _construir_do_zero() -> pd.DataFrame:
    """Lê os CSVs brutos, faz join e retorna DataFrame consolidado."""
    raw_zip = DATA_DIR / "startup-investments.zip"

    if raw_zip.exists() and not RAW_DIR.exists():
        logger.info("Descompactando %s...", raw_zip.name)
        RAW_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(raw_zip, "r") as z:
            z.extractall(RAW_DIR)

    # Tentar localizar os CSVs
    companies_path = _encontrar_csv(RAW_DIR, "companies")
    rounds_path    = _encontrar_csv(RAW_DIR, "rounds")

    if companies_path is None or rounds_path is None:
        raise FileNotFoundError(
            "CSVs não encontrados em data/raw/.\n"
            "Execute: kaggle datasets download -d justinas/startup-investments\n"
            "e coloque o ZIP em data/ ou extraia em data/raw/"
        )

    logger.info("Lendo companies.csv...")
    companies = pd.read_csv(
        companies_path,
        usecols=lambda c: c in COLUNAS_UTEIS,
        encoding="utf-8",
        low_memory=False,
    )

    logger.info("Lendo rounds.csv...")
    rounds = pd.read_csv(
        rounds_path,
        usecols=lambda c: c in COLUNAS_ROUNDS,
        encoding="utf-8",
        low_memory=False,
    )

    df = rounds.merge(
        companies,
        left_on="company_permalink",
        right_on="permalink",
        how="left",
    )

    logger.info("Dataset bruto: %d linhas × %d colunas", *df.shape)
    return df


def _encontrar_csv(diretorio: Path, prefixo: str) -> Path | None:
    """Busca recursivamente por um CSV cujo nome contém o prefixo."""
    matches = list(diretorio.rglob(f"*{prefixo}*.csv"))
    return matches[0] if matches else None
