"""
Geração do painel analítico e gráficos da simulação Monte Carlo.

Saídas:
    - Painel de texto no terminal
    - Histograma da distribuição de retornos (PNG)
    - Curva CDF dos retornos (PNG)
    - CSV com série completa de retornos simulados
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monte_carlo import ResultadoMC

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_SEP = "═" * 52


def exibir_painel(res: ResultadoMC) -> None:
    """Imprime painel analítico no terminal."""
    ct = res.capital_total
    print(f"\n{_SEP}")
    print(f"  SIMULAÇÃO MONTE CARLO — PORTFÓLIO VC")
    print(f"  Iterações: {res.iteracoes:,}  |  Startups: {res.n_startups:,}")
    print(_SEP)
    print(f"  Capital Total Investido        ${ct/1e6:>10.1f} M")
    print(_SEP)
    print(f"  Valor Esperado (Média)         ${res.valor_esperado/1e6:>10.1f} M")
    print(f"  Mediana                        ${res.mediana/1e6:>10.1f} M")
    print(f"  VaR  5%  (pessimista)          ${res.var_5/1e6:>10.1f} M")
    print(f"  VaR 95%  (otimista)            ${res.var_95/1e6:>10.1f} M")
    print(_SEP)
    print(f"  Múltiplo Mediano               {res.multiplicador_med:>10.2f}×")
    print(f"  P(retorno ≥ 3×)                {res.prob_3x*100:>9.1f}%")
    print(f"  P(retorno ≥ 10×)               {res.prob_10x*100:>9.1f}%")
    print(f"{_SEP}\n")


def gerar_histograma(res: ResultadoMC, caminho: Path | None = None) -> Path:
    """Histograma da distribuição de retornos do portfólio."""
    caminho = caminho or OUTPUT_DIR / "histograma_retornos.png"

    retornos_m = res.retornos / 1e6
    ct_m = res.capital_total / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(retornos_m, bins=100, color="#2E86AB", edgecolor="none", alpha=0.85)

    ax.axvline(res.mediana / 1e6,    color="#E84855", lw=2, label=f"Mediana: ${res.mediana/1e6:.1f}M")
    ax.axvline(res.var_5 / 1e6,     color="#F4A261", lw=1.5, ls="--", label=f"VaR 5%: ${res.var_5/1e6:.1f}M")
    ax.axvline(res.var_95 / 1e6,    color="#52B788", lw=1.5, ls="--", label=f"VaR 95%: ${res.var_95/1e6:.1f}M")
    ax.axvline(ct_m,                 color="white",   lw=1.5, ls=":", label=f"Capital: ${ct_m:.1f}M")

    ax.set_xlabel("Retorno do Portfólio (USD Milhões)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição de Retornos — {res.iteracoes:,} iterações Monte Carlo")
    ax.legend(framealpha=0.3)
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#16213E")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

    plt.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Histograma salvo: %s", caminho)
    return caminho


def gerar_cdf(res: ResultadoMC, caminho: Path | None = None) -> Path:
    """Curva CDF dos retornos simulados."""
    caminho = caminho or OUTPUT_DIR / "cdf_retornos.png"

    retornos_sorted = np.sort(res.retornos) / 1e6
    cdf = np.linspace(0, 1, len(retornos_sorted))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(retornos_sorted, cdf, color="#2E86AB", lw=2)

    ax.axhline(0.05, color="#F4A261", lw=1.2, ls="--", label="5% (VaR pessimista)")
    ax.axhline(0.50, color="#E84855", lw=1.2, ls="--", label="50% (Mediana)")
    ax.axhline(0.95, color="#52B788", lw=1.2, ls="--", label="95% (VaR otimista)")
    ax.axvline(res.capital_total / 1e6, color="white", lw=1, ls=":", label="Capital investido")

    ax.set_xlabel("Retorno do Portfólio (USD Milhões)")
    ax.set_ylabel("Probabilidade Acumulada")
    ax.set_title("Função de Distribuição Acumulada (CDF) — Monte Carlo")
    ax.legend(framealpha=0.3)
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#16213E")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

    plt.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("CDF salva: %s", caminho)
    return caminho


def exportar_csv(res: ResultadoMC, caminho: Path | None = None) -> Path:
    """Exporta os retornos simulados e estatísticas para CSV."""
    caminho = caminho or OUTPUT_DIR / "resultados_mc.csv"

    pd.DataFrame({
        "iteracao":          np.arange(1, res.iteracoes + 1),
        "retorno_usd":       res.retornos,
        "multiplicador":     res.retornos / res.capital_total,
    }).to_csv(caminho, index=False)

    logger.info("CSV exportado: %s (%d linhas)", caminho, res.iteracoes)
    return caminho


def gerar_relatorio_completo(res: ResultadoMC) -> None:
    """Exibe painel e gera todos os artefatos gráficos e CSV."""
    exibir_painel(res)
    gerar_histograma(res)
    gerar_cdf(res)
    exportar_csv(res)
    print(f"Gráficos e CSV salvos em: {OUTPUT_DIR}/\n")
