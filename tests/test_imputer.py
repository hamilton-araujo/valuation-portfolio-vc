"""Testes da imputação Ridge — R² > 0 e cobertura de NaNs."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from imputer import imputar, _treinar_modelo, _preparar_features, MIN_AMOSTRAS


def _df_imputacao(n: int = 200, frac_nan: float = 0.3, seed: int = 7) -> pd.DataFrame:
    """DataFrame sintético com valuation parcialmente faltante."""
    rng = np.random.default_rng(seed)
    n_nan = int(n * frac_nan)

    estagios = pd.Categorical(
        rng.choice(["Seed", "Series A", "Series B", "Series C"], size=n),
        categories=["Seed", "Series A", "Series B", "Series C"],
        ordered=True,
    )
    raised = rng.lognormal(14, 1.5, n)
    valuation = raised * rng.uniform(2, 8, n)

    val_com_nan = valuation.copy().astype(float)
    val_com_nan[rng.choice(n, n_nan, replace=False)] = np.nan

    return pd.DataFrame({
        "estagio":          estagios,
        "estagio_num":      pd.Categorical(estagios).codes.astype("int8"),
        "raised_amount_usd": raised,
        "log_raised":       np.log1p(raised),
        "prob_falha":       [0.7 if e == "Seed" else 0.5 for e in estagios],
        "pais":             rng.choice(["USA", "GBR", "DEU", "BRA"], size=n),
        "setor":            rng.choice(["Fintech", "SaaS", "Biotech"], size=n),
        "ano_funding":      rng.integers(2005, 2020, size=n).astype("int16"),
        "valuation":        val_com_nan,
    })


class TestRidgeR2:
    def test_r2_positivo_global(self):
        df = _df_imputacao(300, frac_nan=0.2)
        _, metricas = _treinar_modelo(df)
        assert metricas["r2"] > 0, f"R² esperado > 0, obtido: {metricas['r2']:.4f}"

    def test_mae_finito(self):
        df = _df_imputacao(300, frac_nan=0.2)
        _, metricas = _treinar_modelo(df)
        assert np.isfinite(metricas["mae"])

    def test_r2_por_setor_positivo(self):
        """Ridge por setor deve ter R² > 0 quando há dados suficientes."""
        rng = np.random.default_rng(42)
        n = 400
        raised = rng.lognormal(14, 1.0, n)
        valuation = raised * rng.uniform(3, 7, n)
        val_nan = valuation.copy()
        val_nan[rng.choice(n, 80, replace=False)] = np.nan

        df = pd.DataFrame({
            "estagio":     pd.Categorical(
                rng.choice(["Seed", "Series A"], size=n),
                categories=["Seed", "Series A", "Series B", "Series C"],
                ordered=True,
            ),
            "estagio_num":  np.zeros(n, dtype="int8"),
            "log_raised":   np.log1p(raised),
            "raised_amount_usd": raised,
            "pais":         rng.choice(["USA", "GBR"], size=n),
            "setor":        ["Fintech"] * n,  # setor único com n suficiente
            "ano_funding":  rng.integers(2010, 2020, n).astype("int16"),
            "prob_falha":   np.full(n, 0.5),
            "valuation":    val_nan,
        })

        df_imp = imputar(df)
        assert "valuation_imputado" in df_imp.columns
        assert df_imp["valuation_imputado"].notna().all()


class TestCobertura:
    def test_nenhum_nan_restante(self):
        df = _df_imputacao(200, frac_nan=0.4)
        df_imp = imputar(df)
        assert df_imp["valuation_imputado"].notna().all()

    def test_valores_conhecidos_preservados(self):
        """Startups com valuation real não devem ter valor alterado."""
        df = _df_imputacao(200, frac_nan=0.3)
        mask_val = df["valuation"].notna()
        df_imp = imputar(df)
        # Valores originais devem estar iguais nos registros com valuation real
        np.testing.assert_allclose(
            df_imp.loc[mask_val, "valuation_imputado"].values,
            df.loc[mask_val, "valuation"].values,
        )

    def test_imputados_positivos(self):
        df = _df_imputacao(200, frac_nan=0.5)
        df_imp = imputar(df)
        assert (df_imp["valuation_imputado"] > 0).all()

    def test_sem_valuation_coluna(self):
        """Quando não há coluna 'valuation', função não deve lançar exceção."""
        df = _df_imputacao(200)
        df = df.drop(columns=["valuation"])
        df_imp = imputar(df)
        assert df_imp is not None
