"""
Otimizador de alocação por estágio para portfólio VC.

Por que existe:
    A pergunta de US$ 50M dos LPs: "qual % do fundo vai pra Seed,
    Series A, B, C?". A resposta certa não é intuição — é a alocação
    que maximiza retorno ajustado a risco de ruína.

    Usamos o **Sortino Ratio** (e não Sharpe) porque LPs não se importam
    com volatilidade pra cima — só com perda. Sortino penaliza apenas
    desvio negativo (downside), métrica padrão em VC/PE.

Estratégia:
    Grid search em pesos (Seed, A, B, C) com soma=1, granularidade 10%.
    Para cada combinação válida, sub-amostra do dataset proporcionalmente
    e simula. Devolve top-k alocações ranqueadas por Sortino.
"""

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from monte_carlo import simular

logger = logging.getLogger(__name__)

ESTAGIOS = ["Seed", "Series A", "Series B", "Series C"]


@dataclass
class AlocacaoResultado:
    pesos:         dict[str, float]
    n_startups:    int
    valor_esperado: float
    mediana:       float
    var_5:         float
    sortino:       float
    prob_3x:       float
    prob_perda:    float


def otimizar(
    df: pd.DataFrame,
    n_startups_alvo: int = 30,
    granularidade: float = 0.10,
    iteracoes: int = 5_000,
    investimento_por_startup: float = 1_000_000,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Grid search de alocações por estágio.

    Args:
        df:             DataFrame limpo (todas as startups disponíveis).
        n_startups_alvo: Tamanho do portfólio simulado.
        granularidade:  Passo do grid (0.10 = 10% de increment).
        iteracoes:      Iterações Monte Carlo por alocação.
        top_k:          Quantas top-alocações reportar.

    Returns:
        DataFrame ranqueado por Sortino com pesos e métricas.
    """
    grid = list(_gerar_grid(granularidade))
    logger.info("Grid de alocações: %d combinações × %d iterações",
                len(grid), iteracoes)

    resultados: list[AlocacaoResultado] = []

    for pesos in grid:
        portfolio = _amostrar(df, pesos, n_startups_alvo)
        if len(portfolio) < 4:
            continue

        res = simular(portfolio,
                      iteracoes=iteracoes,
                      investimento_por_startup=investimento_por_startup)

        sortino = _sortino(res.retornos, res.capital_total)
        prob_perda = float((res.retornos < res.capital_total).mean())

        resultados.append(AlocacaoResultado(
            pesos=dict(zip(ESTAGIOS, pesos)),
            n_startups=len(portfolio),
            valor_esperado=res.valor_esperado,
            mediana=res.mediana,
            var_5=res.var_5,
            sortino=sortino,
            prob_3x=res.prob_3x,
            prob_perda=prob_perda,
        ))

    df_rank = pd.DataFrame([
        {
            **{f"w_{e}": r.pesos[e] for e in ESTAGIOS},
            "n_startups":     r.n_startups,
            "valor_esperado": r.valor_esperado,
            "mediana":        r.mediana,
            "var_5":          r.var_5,
            "sortino":        r.sortino,
            "prob_3x":        r.prob_3x,
            "prob_perda":     r.prob_perda,
        }
        for r in resultados
    ]).sort_values("sortino", ascending=False).head(top_k).reset_index(drop=True)

    return df_rank


# ─────────────────────────────────────────────────────────
# Internos
# ─────────────────────────────────────────────────────────

def _gerar_grid(passo: float):
    """Gera tuplas (w_seed, w_a, w_b, w_c) com soma=1."""
    n = int(round(1.0 / passo))
    for combo in itertools.product(range(n + 1), repeat=4):
        if sum(combo) == n:
            yield tuple(c / n for c in combo)


def _amostrar(df: pd.DataFrame, pesos: tuple[float, ...], n_alvo: int) -> pd.DataFrame:
    """Sub-amostra n_alvo startups respeitando pesos por estágio."""
    pedacos = []
    for estagio, peso in zip(ESTAGIOS, pesos):
        n = int(round(peso * n_alvo))
        if n == 0:
            continue
        pool = df[df["estagio"] == estagio]
        if len(pool) == 0:
            continue
        pedacos.append(pool.sample(n=min(n, len(pool)),
                                    replace=len(pool) < n,
                                    random_state=42))
    if not pedacos:
        return df.iloc[0:0]
    return pd.concat(pedacos, ignore_index=True)


def _sortino(retornos: np.ndarray, capital: float, alvo: float = 1.0) -> float:
    """
    Sortino Ratio adaptado a VC.

    Numerador  : (média do múltiplo) - alvo (default 1× = recuperar capital).
    Denominador: desvio padrão dos retornos abaixo do alvo (downside dev).
    """
    multiplos = retornos / capital
    excesso = multiplos - alvo
    downside = multiplos[multiplos < alvo] - alvo
    if len(downside) < 2:
        return float("inf") if excesso.mean() > 0 else 0.0
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return 0.0
    return float(excesso.mean() / dd)
