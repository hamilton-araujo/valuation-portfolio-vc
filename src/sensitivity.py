"""
Análise de sensibilidade — Tornado Chart das premissas do modelo VC.

Por que existe:
    Modelos quantitativos morrem quando o IC pergunta: "se a P(falha) Seed
    for 80% em vez de 70%, o portfólio ainda vira 3×?". Este módulo varia
    cada premissa em ±15% (ceteris paribus) e mede a amplitude do impacto
    no múltiplo mediano. O resultado é o tornado clássico: barras
    horizontais ordenadas por sensibilidade, expondo as 3-4 premissas
    que realmente movem a tese.

Parâmetros testados:
    - P(falha) por estágio (Seed / A / B / C)
    - μ (drift) do log-normal por estágio
    - σ (dispersão) do log-normal por estágio
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from monte_carlo import LOGNORM_PARAMS
from scenarios import _simular_com_params

logger = logging.getLogger(__name__)


@dataclass
class SensibilidadeItem:
    parametro:    str
    valor_baixo:  float
    valor_alto:   float
    multiplo_baixo: float   # múltiplo mediano com choque pra baixo
    multiplo_alto:  float
    amplitude:    float     # |alto - baixo|, ordena tornado


VARIACAO = 0.15  # ±15% em torno do valor base


def analisar(
    df: pd.DataFrame,
    iteracoes: int = 5_000,
    investimento_por_startup: float = 1_000_000,
) -> pd.DataFrame:
    """
    Roda análise de sensibilidade ceteris paribus.

    Returns:
        DataFrame com 12 linhas (4 estágios × 3 params) ordenado por amplitude.
    """
    base_params = dict(LOGNORM_PARAMS)
    base_pf = df["prob_falha"].copy()

    # Múltiplo mediano de referência
    res_base = _simular_com_params(df.copy(), base_params,
                                    iteracoes, investimento_por_startup)
    mult_base = res_base.multiplicador_med
    logger.info("Múltiplo mediano base: %.2fx", mult_base)

    itens: list[SensibilidadeItem] = []

    for estagio in ["Seed", "Series A", "Series B", "Series C"]:
        # ── P(falha) ────────────────────────────────────────────────
        pf_base = df.loc[df["estagio"] == estagio, "prob_falha"].iloc[0] \
                  if (df["estagio"] == estagio).any() else 0.5

        for delta_label, delta in [("baixo", -VARIACAO), ("alto", +VARIACAO)]:
            df_mod = df.copy()
            mask = df_mod["estagio"] == estagio
            df_mod.loc[mask, "prob_falha"] = (df_mod.loc[mask, "prob_falha"] + delta).clip(0.05, 0.95)
            res = _simular_com_params(df_mod, base_params, iteracoes, investimento_por_startup)
            if delta_label == "baixo":
                m_low, v_low = res.multiplicador_med, float((pf_base + delta).clip(0.05, 0.95))
            else:
                m_high, v_high = res.multiplicador_med, float((pf_base + delta).clip(0.05, 0.95))

        itens.append(SensibilidadeItem(
            parametro=f"P(falha) {estagio}",
            valor_baixo=v_low, valor_alto=v_high,
            multiplo_baixo=m_low, multiplo_alto=m_high,
            amplitude=abs(m_high - m_low),
        ))

        # ── μ (drift) ───────────────────────────────────────────────
        mu_base, sig_base = base_params[estagio]
        for delta_label, delta in [("baixo", -VARIACAO), ("alto", +VARIACAO)]:
            params_mod = dict(base_params)
            params_mod[estagio] = (mu_base + delta, sig_base)
            res = _simular_com_params(df.copy(), params_mod, iteracoes, investimento_por_startup)
            if delta_label == "baixo":
                m_low, v_low = res.multiplicador_med, mu_base + delta
            else:
                m_high, v_high = res.multiplicador_med, mu_base + delta

        itens.append(SensibilidadeItem(
            parametro=f"μ {estagio}",
            valor_baixo=v_low, valor_alto=v_high,
            multiplo_baixo=m_low, multiplo_alto=m_high,
            amplitude=abs(m_high - m_low),
        ))

        # ── σ (dispersão) ───────────────────────────────────────────
        for delta_label, delta in [("baixo", -VARIACAO), ("alto", +VARIACAO)]:
            params_mod = dict(base_params)
            params_mod[estagio] = (mu_base, sig_base * (1 + delta))
            res = _simular_com_params(df.copy(), params_mod, iteracoes, investimento_por_startup)
            if delta_label == "baixo":
                m_low, v_low = res.multiplicador_med, sig_base * (1 + delta)
            else:
                m_high, v_high = res.multiplicador_med, sig_base * (1 + delta)

        itens.append(SensibilidadeItem(
            parametro=f"σ {estagio}",
            valor_baixo=v_low, valor_alto=v_high,
            multiplo_baixo=m_low, multiplo_alto=m_high,
            amplitude=abs(m_high - m_low),
        ))

    df_out = pd.DataFrame([i.__dict__ for i in itens]) \
               .sort_values("amplitude", ascending=False) \
               .reset_index(drop=True)
    df_out["multiplo_base"] = mult_base
    return df_out
