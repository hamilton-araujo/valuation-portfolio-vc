"""Testes do motor Monte Carlo — ordenamento VaR e invariantes estatísticos."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monte_carlo import simular, ResultadoMC


def _df_portfolio(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Portfólio sintético mínimo para simulação."""
    rng = np.random.default_rng(seed)
    estagios = pd.Categorical(
        rng.choice(["Seed", "Series A", "Series B", "Series C"], size=n),
        categories=["Seed", "Series A", "Series B", "Series C"],
        ordered=True,
    )
    prob_falha = np.where(
        estagios == "Seed", 0.70,
        np.where(estagios == "Series A", 0.50,
        np.where(estagios == "Series B", 0.35, 0.25))
    )
    return pd.DataFrame({
        "estagio":    estagios,
        "prob_falha": prob_falha.astype("float32"),
    })


class TestOrdenamentoVar:
    def test_var5_menor_mediana_menor_var95(self):
        df = _df_portfolio(100)
        res = simular(df, iteracoes=5_000, seed=42)
        assert res.var_5 < res.mediana, "VaR 5% deve ser < Mediana"
        assert res.mediana < res.var_95, "Mediana deve ser < VaR 95%"

    def test_var5_menor_valor_esperado(self):
        df = _df_portfolio(100)
        res = simular(df, iteracoes=5_000, seed=42)
        assert res.var_5 < res.valor_esperado

    def test_var5_positivo(self):
        """Mesmo no pior cenário o portfólio pode ter algum retorno."""
        df = _df_portfolio(50)
        res = simular(df, iteracoes=5_000, seed=7)
        assert res.var_5 >= 0


class TestInvariantesEstatisticos:
    def test_n_startups_correto(self):
        df = _df_portfolio(73)
        res = simular(df, iteracoes=1_000, seed=1)
        assert res.n_startups == 73

    def test_iteracoes_corretas(self):
        df = _df_portfolio(30)
        res = simular(df, iteracoes=2_000, seed=2)
        assert len(res.retornos) == 2_000

    def test_capital_total_correto(self):
        df = _df_portfolio(50)
        investimento = 500_000
        res = simular(df, iteracoes=1_000, investimento_por_startup=investimento, seed=3)
        assert res.capital_total == 50 * investimento

    def test_retornos_nao_negativos(self):
        df = _df_portfolio(80)
        res = simular(df, iteracoes=3_000, seed=5)
        assert (res.retornos >= 0).all()

    def test_prob_3x_prob_10x_ordenadas(self):
        df = _df_portfolio(100)
        res = simular(df, iteracoes=5_000, seed=9)
        assert res.prob_3x >= res.prob_10x

    def test_probabilidades_entre_0_e_1(self):
        df = _df_portfolio(60)
        res = simular(df, iteracoes=3_000, seed=11)
        assert 0 <= res.prob_3x <= 1
        assert 0 <= res.prob_10x <= 1

    def test_reproducibilidade(self):
        df = _df_portfolio(50)
        res1 = simular(df, iteracoes=1_000, seed=42)
        res2 = simular(df, iteracoes=1_000, seed=42)
        np.testing.assert_array_equal(res1.retornos, res2.retornos)

    def test_seed_diferente_resultado_diferente(self):
        df = _df_portfolio(50)
        res1 = simular(df, iteracoes=1_000, seed=1)
        res2 = simular(df, iteracoes=1_000, seed=2)
        assert not np.array_equal(res1.retornos, res2.retornos)


class TestCasosFronteira:
    def test_portfolio_vazio_levanta_erro(self):
        df = pd.DataFrame({"estagio": [], "prob_falha": []})
        with pytest.raises((ValueError, Exception)):
            simular(df, iteracoes=100)

    def test_startup_unica(self):
        df = pd.DataFrame({
            "estagio":    pd.Categorical(["Seed"], categories=["Seed", "Series A", "Series B", "Series C"]),
            "prob_falha": [0.70],
        })
        res = simular(df, iteracoes=2_000, seed=0)
        assert res.n_startups == 1
        assert res.var_5 <= res.var_95

    def test_probabilidade_falha_zero(self):
        """Com prob_falha=0, todos os retornos devem ser > 0."""
        df = pd.DataFrame({
            "estagio":    pd.Categorical(["Series C"] * 10, categories=["Seed", "Series A", "Series B", "Series C"]),
            "prob_falha": [0.0] * 10,
        })
        res = simular(df, iteracoes=1_000, seed=0)
        assert (res.retornos > 0).all()
