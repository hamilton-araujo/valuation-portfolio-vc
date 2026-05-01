"""
Motor de simulação Monte Carlo vetorizado para portfólio de Venture Capital.

Modelo:
    Para cada startup e cada iteração:
        1. Sobrevivência ~ Binomial(1, 1 - prob_falha)
        2. Multiplicador  ~ LogNormal(μ, σ)  |  se sobreviveu
        3. Retorno = investimento × sobrevivência × multiplicador
    Retorno do portfólio = Σ retornos individuais

Parâmetros LogNormal calibrados por estágio (literatura VC):
    Seed:     μ=1.0, σ=1.5  (maior upside, maior dispersão)
    Series A: μ=0.9, σ=1.3
    Series B: μ=0.7, σ=1.1
    Series C: μ=0.5, σ=0.9
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Parâmetros Log-Normal por estágio (μ e σ do log do multiplicador)
LOGNORM_PARAMS = {
    "Seed":     (1.0, 1.5),
    "Series A": (0.9, 1.3),
    "Series B": (0.7, 1.1),
    "Series C": (0.5, 0.9),
}

DEFAULT_INVESTIMENTO_USD = 1_000_000  # $1M por startup


@dataclass
class ResultadoMC:
    """Resultados da simulação Monte Carlo."""
    iteracoes:         int
    n_startups:        int
    capital_total:     float
    retornos:          np.ndarray   # shape (iteracoes,)
    valor_esperado:    float
    mediana:           float
    var_5:             float        # VaR 5% pessimista
    var_95:            float        # VaR 95% otimista
    prob_3x:           float
    prob_10x:          float
    multiplicador_med: float        # mediana do múltiplo total


def simular(
    df: pd.DataFrame,
    iteracoes: int = 10_000,
    investimento_por_startup: float = DEFAULT_INVESTIMENTO_USD,
    seed: int = 42,
) -> ResultadoMC:
    """
    Executa a simulação Monte Carlo sobre o portfólio.

    Args:
        df:                       DataFrame limpo com colunas estagio e prob_falha.
        iteracoes:                Número de simulações (10k–100k recomendado).
        investimento_por_startup: Capital alocado por startup em USD.
        seed:                     Semente aleatória para reprodutibilidade.

    Returns:
        ResultadoMC com estatísticas do portfólio simulado.
    """
    rng = np.random.default_rng(seed)

    # Garantir que temos estagio e prob_falha
    df = _validar_df(df)

    estagios   = df["estagio"].astype(str).values       # (n,)
    prob_falha = df["prob_falha"].values.astype(float)  # (n,)
    n = len(df)
    capital_total = n * investimento_por_startup

    logger.info("Simulando %d iterações × %d startups...", iteracoes, n)

    # ── Sobrevivência: Binomial vetorizada ──────────────────────────────────
    # shape (iteracoes, n)
    prob_sobrev = 1.0 - prob_falha
    sobreviveu = rng.random((iteracoes, n)) < prob_sobrev  # bool array

    # ── Multiplicadores: Log-Normal vetorizado por estágio ──────────────────
    mu_arr  = np.array([LOGNORM_PARAMS.get(e, (0.5, 1.0))[0] for e in estagios])
    sig_arr = np.array([LOGNORM_PARAMS.get(e, (0.5, 1.0))[1] for e in estagios])

    # z ~ N(0,1), shape (iteracoes, n)
    z = rng.standard_normal((iteracoes, n))
    multiplicadores = np.exp(mu_arr + sig_arr * z)  # Log-Normal

    # ── Retorno do portfólio por iteração ───────────────────────────────────
    # (iteracoes, n) → soma por iteração → (iteracoes,)
    retornos_startup = sobreviveu * multiplicadores * investimento_por_startup
    retornos_portfolio = retornos_startup.sum(axis=1)

    # ── Estatísticas ────────────────────────────────────────────────────────
    var_5  = float(np.percentile(retornos_portfolio, 5))
    var_95 = float(np.percentile(retornos_portfolio, 95))

    return ResultadoMC(
        iteracoes=iteracoes,
        n_startups=n,
        capital_total=capital_total,
        retornos=retornos_portfolio,
        valor_esperado=float(retornos_portfolio.mean()),
        mediana=float(np.median(retornos_portfolio)),
        var_5=var_5,
        var_95=var_95,
        prob_3x=float((retornos_portfolio >= 3 * capital_total).mean()),
        prob_10x=float((retornos_portfolio >= 10 * capital_total).mean()),
        multiplicador_med=float(np.median(retornos_portfolio) / capital_total),
    )


def _validar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas mínimas necessárias."""
    df = df.copy()

    if "prob_falha" not in df.columns:
        from cleaner import PROB_FALHA
        df["prob_falha"] = df["estagio"].map(PROB_FALHA).fillna(0.5)

    df = df[df["prob_falha"].notna() & df["estagio"].notna()]
    if len(df) == 0:
        raise ValueError("DataFrame vazio após validação — verifique colunas estagio/prob_falha.")

    return df
