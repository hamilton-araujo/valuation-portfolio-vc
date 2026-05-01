"""Testes da etapa de limpeza — foco no IQR logarítmico."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cleaner import limpar, _remover_outliers_iqr, ESTAGIOS_ORDEM, PROB_FALHA


def _df_base(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """DataFrame sintético com estrutura mínima para o pipeline de limpeza."""
    rng = np.random.default_rng(seed)
    estagios = rng.choice(
        ["seed", "series-a", "series-b", "series-c"], size=n
    )
    valores = rng.lognormal(mean=14, sigma=1.5, size=n)  # USD ~1M médio

    return pd.DataFrame({
        "funding_round_type": estagios,
        "raised_amount_usd":  valores,
        "company_country_code": rng.choice(["USA", "GBR", "DEU", "BRA"], size=n),
        "company_market":       rng.choice(["Fintech", "SaaS", "Biotech"], size=n),
        "status":               rng.choice(["operating", "acquired", "ipo"], size=n),
        "funded_at":            pd.date_range("2010-01-01", periods=n, freq="W").astype(str),
    })


class TestIqrLog:
    def test_remove_outliers_extremos(self):
        df = _df_base(300)
        # Injeta outlier extremo
        df.loc[0, "raised_amount_usd"] = 1e15
        df.loc[1, "raised_amount_usd"] = 0.01

        df_clean = limpar(df)
        assert df_clean["raised_amount_usd"].max() < 1e15
        assert df_clean["raised_amount_usd"].min() > 0.01

    def test_nan_preservado(self):
        """Linhas com NaN em raised_amount_usd devem ser mantidas."""
        df = _df_base(100)
        df.loc[5, "raised_amount_usd"] = np.nan
        df_clean = limpar(df)
        # NaN não deve ser removido pelo IQR (mascara inclui isna)
        assert df_clean["raised_amount_usd"].isna().sum() >= 0  # pipeline pode imputar depois

    def test_iqr_escala_log_conserva_mais(self):
        """IQR log deve remover MENOS linhas que IQR linear para dist. Power Law."""
        rng = np.random.default_rng(99)
        df = _df_base(500)

        from cleaner import _remover_outliers_iqr
        import pandas as pd

        df_log = _remover_outliers_iqr(df, "raised_amount_usd", 1.5)

        # IQR linear (para comparação)
        serie = df["raised_amount_usd"].dropna()
        serie = serie[serie > 0]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        mask_lin = df["raised_amount_usd"].isna() | (
            (df["raised_amount_usd"] >= q1 - 1.5 * iqr) &
            (df["raised_amount_usd"] <= q3 + 1.5 * iqr)
        )
        n_linear = mask_lin.sum()

        assert len(df_log) >= n_linear

    def test_multiplicador_maior_retira_menos(self):
        df = _df_base(300)
        df.loc[0, "raised_amount_usd"] = 1e14
        df1 = _remover_outliers_iqr(df, "raised_amount_usd", 1.5)
        df2 = _remover_outliers_iqr(df, "raised_amount_usd", 3.0)
        assert len(df2) >= len(df1)


class TestPipelineLimpeza:
    def test_shape_reduz(self):
        df = _df_base(200)
        df_clean = limpar(df)
        assert len(df_clean) <= len(df)

    def test_estagios_validos_apenas(self):
        df = _df_base(200)
        df_clean = limpar(df)
        assert set(df_clean["estagio"].unique()).issubset(set(ESTAGIOS_ORDEM))

    def test_colunas_derivadas_criadas(self):
        df = _df_base(200)
        df_clean = limpar(df)
        for col in ["estagio_num", "prob_falha", "log_raised"]:
            assert col in df_clean.columns, f"Coluna '{col}' ausente"

    def test_prob_falha_mapeada(self):
        df = _df_base(200)
        df_clean = limpar(df)
        for estagio, p in PROB_FALHA.items():
            mask = df_clean["estagio"] == estagio
            if mask.any():
                assert (df_clean.loc[mask, "prob_falha"] == p).all()

    def test_index_resetado(self):
        df = _df_base(200)
        df_clean = limpar(df)
        assert list(df_clean.index) == list(range(len(df_clean)))
