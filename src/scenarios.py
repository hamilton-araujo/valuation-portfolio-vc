"""
Cenários macroeconômicos aplicados ao portfólio VC.

Por que existe:
    LPs não aceitam um único número de retorno esperado. Eles querem saber
    "e se vier uma recessão como 2008?" e "e se for um ciclo como 2021?".
    Este módulo aplica choques estruturais nas premissas do modelo —
    probabilidades de falha e dispersão dos retornos — produzindo três
    versões do portfólio que viram input direto da decisão de capital call.

Cenários:
    BASE      — premissas históricas calibradas (literatura VC).
    RECESSAO  — choque negativo: +20pp na P(falha) por estágio,
                σ comprimido em 30%, μ reduzido em 0.4 (queda de upside).
    BOOM      — ciclo expansionista: -15pp na P(falha), σ +25%, μ +0.3.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from monte_carlo import LOGNORM_PARAMS, ResultadoMC, simular

logger = logging.getLogger(__name__)


@dataclass
class Cenario:
    nome: str
    descricao: str
    delta_prob_falha: float   # adiciona aos p_falha base (clip 0..0.95)
    delta_mu: float            # soma a μ do log-normal por estágio
    fator_sigma: float         # multiplica σ do log-normal


CENARIOS = {
    "base": Cenario(
        nome="Base",
        descricao="Premissas históricas calibradas (literatura VC).",
        delta_prob_falha=0.0, delta_mu=0.0, fator_sigma=1.0,
    ),
    "recessao": Cenario(
        nome="Recessão",
        descricao="Choque 2008-like: mortalidade +20pp, upside comprimido.",
        delta_prob_falha=0.20, delta_mu=-0.4, fator_sigma=0.7,
    ),
    "boom": Cenario(
        nome="Boom",
        descricao="Ciclo 2021-like: mortalidade -15pp, dispersão amplificada.",
        delta_prob_falha=-0.15, delta_mu=0.3, fator_sigma=1.25,
    ),
}


def aplicar_cenario(df: pd.DataFrame, cenario: Cenario) -> tuple[pd.DataFrame, dict]:
    """Devolve cópia do df com prob_falha ajustada + dict de params LogNormal."""
    df = df.copy()
    df["prob_falha"] = (df["prob_falha"] + cenario.delta_prob_falha).clip(0.05, 0.95)

    params = {
        e: (mu + cenario.delta_mu, sig * cenario.fator_sigma)
        for e, (mu, sig) in LOGNORM_PARAMS.items()
    }
    return df, params


def simular_cenarios(
    df: pd.DataFrame,
    iteracoes: int = 10_000,
    investimento_por_startup: float = 1_000_000,
) -> dict[str, ResultadoMC]:
    """Roda Monte Carlo nos três cenários e devolve dict {nome: ResultadoMC}."""
    resultados = {}
    for chave, cen in CENARIOS.items():
        logger.info("Cenário '%s': %s", cen.nome, cen.descricao)
        df_cen, params = aplicar_cenario(df, cen)
        res = _simular_com_params(df_cen, params, iteracoes, investimento_por_startup)
        resultados[chave] = res
    return resultados


def comparar(resultados: dict[str, ResultadoMC]) -> pd.DataFrame:
    """Tabela comparativa entre cenários."""
    rows = []
    for chave, r in resultados.items():
        cen = CENARIOS[chave]
        rows.append({
            "cenario":         cen.nome,
            "valor_esperado":  r.valor_esperado,
            "mediana":         r.mediana,
            "var_5":           r.var_5,
            "var_95":          r.var_95,
            "multiplo_med":    r.multiplicador_med,
            "prob_3x":         r.prob_3x,
            "prob_10x":        r.prob_10x,
            "prob_perda":      float((r.retornos < r.capital_total).mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# Interno — re-simulação parametrizada
# ─────────────────────────────────────────────────────────

def _simular_com_params(
    df: pd.DataFrame,
    params_lognorm: dict,
    iteracoes: int,
    investimento: float,
    seed: int = 42,
) -> ResultadoMC:
    rng = np.random.default_rng(seed)
    estagios = df["estagio"].astype(str).values
    prob_falha = df["prob_falha"].values.astype(float)
    n = len(df)
    capital_total = n * investimento

    sobreviveu = rng.random((iteracoes, n)) < (1.0 - prob_falha)
    mu_arr  = np.array([params_lognorm.get(e, (0.5, 1.0))[0] for e in estagios])
    sig_arr = np.array([params_lognorm.get(e, (0.5, 1.0))[1] for e in estagios])
    z = rng.standard_normal((iteracoes, n))
    multiplicadores = np.exp(mu_arr + sig_arr * z)

    retornos = (sobreviveu * multiplicadores * investimento).sum(axis=1)

    return ResultadoMC(
        iteracoes=iteracoes, n_startups=n, capital_total=capital_total,
        retornos=retornos,
        valor_esperado=float(retornos.mean()),
        mediana=float(np.median(retornos)),
        var_5=float(np.percentile(retornos, 5)),
        var_95=float(np.percentile(retornos, 95)),
        prob_3x=float((retornos >= 3 * capital_total).mean()),
        prob_10x=float((retornos >= 10 * capital_total).mean()),
        multiplicador_med=float(np.median(retornos) / capital_total),
    )
