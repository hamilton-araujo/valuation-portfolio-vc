"""
Tearsheet executivo — relatório markdown + gráficos comparativos para LPs.

Por que existe:
    O CSV de retornos não convence comitê de investimento. Este módulo
    consolida cenários + alocação ótima + sensibilidade num único
    documento markdown com a recomendação final: APROVAR / REVISAR / REJEITAR.

Heurística de decisão (calibrada por benchmarks Cambridge Associates):
    APROVAR  : prob_3x ≥ 30% no cenário base  E  prob_perda ≤ 25% na recessão
    REVISAR  : prob_3x ∈ [15%, 30%)            OU  prob_perda ∈ (25%, 40%]
    REJEITAR : prob_3x < 15% no base           OU  prob_perda > 40% na recessão
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monte_carlo import ResultadoMC

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def gerar(
    res_base: ResultadoMC,
    cenarios: dict[str, ResultadoMC] | None = None,
    df_otimizacao: pd.DataFrame | None = None,
    df_sensibilidade: pd.DataFrame | None = None,
) -> Path:
    """Gera tearsheet markdown consolidado + gráficos auxiliares."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    md_path = OUTPUT_DIR / "tearsheet_executivo.md"

    decisao, justificativa = _recomendar(res_base, cenarios)

    if cenarios:
        _grafico_cenarios(cenarios, OUTPUT_DIR / "cenarios_comparativo.png")
    if df_sensibilidade is not None:
        _grafico_tornado(df_sensibilidade, OUTPUT_DIR / "sensibilidade_tornado.png")

    md = _montar_markdown(res_base, cenarios, df_otimizacao,
                          df_sensibilidade, decisao, justificativa)
    md_path.write_text(md, encoding="utf-8")
    logger.info("Tearsheet salvo: %s", md_path)
    return md_path


# ─────────────────────────────────────────────────────────
# Decisão e justificativa
# ─────────────────────────────────────────────────────────

def _recomendar(
    res_base: ResultadoMC,
    cenarios: dict[str, ResultadoMC] | None,
) -> tuple[str, list[str]]:
    p3x_base = res_base.prob_3x
    prob_perda_rec = None
    if cenarios and "recessao" in cenarios:
        r = cenarios["recessao"]
        prob_perda_rec = float((r.retornos < r.capital_total).mean())

    razoes: list[str] = []

    if prob_perda_rec is not None and prob_perda_rec > 0.40:
        razoes.append(f"Risco de perda em recessão = {prob_perda_rec*100:.1f}% (limite IC: 40%)")
        return "REJEITAR", razoes
    if p3x_base < 0.15:
        razoes.append(f"P(3×) base = {p3x_base*100:.1f}% (mínimo aceitável: 15%)")
        return "REJEITAR", razoes

    if p3x_base >= 0.30 and (prob_perda_rec is None or prob_perda_rec <= 0.25):
        razoes.append(f"P(3×) base = {p3x_base*100:.1f}% ≥ 30%")
        if prob_perda_rec is not None:
            razoes.append(f"Perda em recessão = {prob_perda_rec*100:.1f}% ≤ 25%")
        return "APROVAR", razoes

    razoes.append(f"P(3×) base = {p3x_base*100:.1f}% (zona de revisão)")
    if prob_perda_rec is not None:
        razoes.append(f"Perda em recessão = {prob_perda_rec*100:.1f}%")
    return "REVISAR", razoes


# ─────────────────────────────────────────────────────────
# Gráficos
# ─────────────────────────────────────────────────────────

def _grafico_cenarios(cenarios: dict[str, ResultadoMC], caminho: Path):
    nomes = list(cenarios.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cores = {"base": "#2E86AB", "recessao": "#E84855", "boom": "#52B788"}

    # CDFs sobrepostas
    for nome in nomes:
        r = cenarios[nome]
        x = np.sort(r.retornos) / 1e6
        y = np.linspace(0, 1, len(x))
        axes[0].plot(x, y, lw=2, label=nome.title(), color=cores.get(nome, "gray"))
    axes[0].axvline(cenarios[nomes[0]].capital_total / 1e6, color="black",
                    ls=":", lw=1, label="Capital")
    axes[0].set_xlabel("Retorno do Portfólio (USD M)")
    axes[0].set_ylabel("Probabilidade Acumulada")
    axes[0].set_title("CDF dos Retornos por Cenário")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Barras de métricas-chave
    metricas = ["prob_3x", "prob_10x", "prob_perda"]
    labels = ["P(≥3×)", "P(≥10×)", "P(perda)"]
    x = np.arange(len(metricas))
    w = 0.25
    for i, nome in enumerate(nomes):
        r = cenarios[nome]
        ct = r.capital_total
        vals = [r.prob_3x, r.prob_10x, float((r.retornos < ct).mean())]
        axes[1].bar(x + (i - 1) * w, [v * 100 for v in vals],
                    width=w, label=nome.title(), color=cores.get(nome, "gray"))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Probabilidade (%)")
    axes[1].set_title("Métricas-chave por Cenário")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grafico_tornado(df_sens: pd.DataFrame, caminho: Path):
    df = df_sens.head(8).iloc[::-1]
    base = float(df["multiplo_base"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))

    ax.barh(y, df["multiplo_baixo"] - base, left=base,
            color="#E84855", alpha=0.85, label=f"Choque −15%")
    ax.barh(y, df["multiplo_alto"] - base,  left=base,
            color="#52B788", alpha=0.85, label=f"Choque +15%")

    ax.axvline(base, color="black", lw=1.5, ls="--", label=f"Base: {base:.2f}×")
    ax.set_yticks(y)
    ax.set_yticklabels(df["parametro"])
    ax.set_xlabel("Múltiplo Mediano do Portfólio")
    ax.set_title("Análise de Sensibilidade — Tornado das Premissas")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# Markdown
# ─────────────────────────────────────────────────────────

def _montar_markdown(
    res: ResultadoMC,
    cenarios: dict[str, ResultadoMC] | None,
    df_otim: pd.DataFrame | None,
    df_sens: pd.DataFrame | None,
    decisao: str,
    razoes: list[str],
) -> str:
    badge = {"APROVAR": "✅", "REVISAR": "⚠️", "REJEITAR": "❌"}[decisao]
    lines = [
        "# Tearsheet Executivo — Portfólio VC",
        "",
        f"## Recomendação ao Comitê de Investimento: {badge} **{decisao}**",
        "",
        "**Justificativa:**",
        "",
    ]
    for r in razoes:
        lines.append(f"- {r}")
    lines += [
        "",
        "---",
        "",
        "## 1. Cenário Base — Premissas Históricas",
        "",
        f"- **Capital investido:** USD {res.capital_total/1e6:.1f} M",
        f"- **Startups no portfólio:** {res.n_startups}",
        f"- **Iterações Monte Carlo:** {res.iteracoes:,}",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Valor esperado | USD {res.valor_esperado/1e6:.1f} M |",
        f"| Mediana | USD {res.mediana/1e6:.1f} M |",
        f"| Múltiplo mediano | {res.multiplicador_med:.2f}× |",
        f"| VaR 5% (pessimista) | USD {res.var_5/1e6:.1f} M |",
        f"| VaR 95% (otimista) | USD {res.var_95/1e6:.1f} M |",
        f"| P(retorno ≥ 3×) | {res.prob_3x*100:.1f}% |",
        f"| P(retorno ≥ 10×) | {res.prob_10x*100:.1f}% |",
        "",
    ]

    if cenarios:
        lines += [
            "## 2. Stress-Test Macro",
            "",
            "Choque comparativo entre Base, Recessão (2008-like) e Boom (2021-like).",
            "",
            "![Cenários](cenarios_comparativo.png)",
            "",
            "| Cenário | Mediana (M USD) | Múltiplo | P(3×) | P(perda) |",
            "|---|---|---|---|---|",
        ]
        for nome, r in cenarios.items():
            ct = r.capital_total
            p_perda = float((r.retornos < ct).mean())
            lines.append(
                f"| {nome.title()} | {r.mediana/1e6:.1f} | {r.multiplicador_med:.2f}× "
                f"| {r.prob_3x*100:.1f}% | {p_perda*100:.1f}% |"
            )
        lines.append("")

    if df_otim is not None and len(df_otim):
        top = df_otim.iloc[0]
        lines += [
            "## 3. Alocação Ótima por Estágio",
            "",
            "Grid search maximizando **Sortino Ratio** (downside-only).",
            "",
            "**Top alocação:**",
            "",
            f"- Seed: **{top['w_Seed']*100:.0f}%**",
            f"- Series A: **{top['w_Series A']*100:.0f}%**",
            f"- Series B: **{top['w_Series B']*100:.0f}%**",
            f"- Series C: **{top['w_Series C']*100:.0f}%**",
            "",
            f"Sortino = **{top['sortino']:.3f}** · P(3×) = {top['prob_3x']*100:.1f}% "
            f"· P(perda) = {top['prob_perda']*100:.1f}%",
            "",
        ]

    if df_sens is not None and len(df_sens):
        top3 = df_sens.head(3)
        lines += [
            "## 4. Sensibilidade às Premissas",
            "",
            "Tornado chart: variação ±15% ceteris paribus em cada premissa.",
            "",
            "![Tornado](sensibilidade_tornado.png)",
            "",
            "**Top 3 premissas mais sensíveis:**",
            "",
        ]
        for _, row in top3.iterrows():
            lines.append(
                f"- **{row['parametro']}** — amplitude {row['amplitude']:.2f}× "
                f"(de {row['multiplo_baixo']:.2f}× a {row['multiplo_alto']:.2f}×)"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Metodologia",
        "",
        "- **Mortalidade:** Binomial(1, p_falha) calibrada por estágio (literatura VC).",
        "- **Multiplicador de retorno:** LogNormal(μ, σ) — captura *power law* característica.",
        "- **Imputação de valuations faltantes:** Ridge Regression segmentada por setor (K-Fold CV).",
        "- **Decisão:** heurística calibrada por benchmarks Cambridge Associates (mediana global VC ≈ 2.0× TVPI).",
        "",
    ]
    return "\n".join(lines)
