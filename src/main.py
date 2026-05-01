"""
CLI principal — Valuation e Simulação de Portfólio de Venture Capital.

Modos:
    single       — Monte Carlo único (cenário base) + painel + gráficos.
    scenarios    — Compara Base / Recessão / Boom.
    optimize     — Grid search da alocação ótima por estágio (Sortino).
    sensitivity  — Tornado de sensibilidade das premissas.
    full         — Roda tudo + tearsheet executivo com decisão IC.

Exemplos:
    python src/main.py --mode single --iterations 10000
    python src/main.py --mode full --iterations 10000
    python src/main.py --mode optimize --target 30 --grid 0.10
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest
import cleaner
import imputer
import monte_carlo
import report
import scenarios
import optimizer
import sensitivity
import tearsheet


def _configurar_log(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulação Monte Carlo de Portfólio de Venture Capital.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["single", "scenarios", "optimize", "sensitivity", "full"],
                   default="single", help="Modo de execução (padrão: single).")
    p.add_argument("--iterations", type=int, default=10_000, metavar="N")
    p.add_argument("--confidence", type=float, default=0.95, metavar="C")
    p.add_argument("--stages", nargs="+", default=None, metavar="ESTAGIO")
    p.add_argument("--sectors", nargs="+", default=None, metavar="SETOR")
    p.add_argument("--capital", type=float, default=1_000_000, metavar="USD")
    p.add_argument("--iqr-multiplier", type=float, default=1.5, metavar="M")
    p.add_argument("--target", type=int, default=30, metavar="N",
                   help="(optimize) tamanho alvo do portfólio.")
    p.add_argument("--grid", type=float, default=0.10, metavar="G",
                   help="(optimize) granularidade do grid.")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--force-reload", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _validar_args(args: argparse.Namespace) -> None:
    if args.iterations < 100:
        raise ValueError(f"--iterations deve ser ≥ 100 (recebido: {args.iterations}).")
    if not 0 < args.confidence < 1:
        raise ValueError(f"--confidence deve estar em (0, 1).")
    if args.capital <= 0:
        raise ValueError(f"--capital deve ser positivo.")
    if args.iqr_multiplier <= 0:
        raise ValueError(f"--iqr-multiplier deve ser positivo.")


def _preparar_df(args):
    df = ingest.carregar(force_reload=args.force_reload)
    df = cleaner.limpar(df, iqr_multiplier=args.iqr_multiplier)

    if args.stages:
        df = df[df["estagio"].isin(args.stages)]
    if args.sectors and "setor" in df.columns:
        df = df[df["setor"].isin(args.sectors)]

    if len(df) == 0:
        print("Nenhuma startup restante após filtros.", file=sys.stderr)
        sys.exit(1)
    df = imputer.imputar(df)
    return df


def main() -> None:
    args = _parse_args()
    _configurar_log(args.verbose)

    try:
        _validar_args(args)
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)

    df = _preparar_df(args)

    if args.mode == "single":
        res = monte_carlo.simular(df, iteracoes=args.iterations,
                                  investimento_por_startup=args.capital)
        if args.no_report:
            report.exibir_painel(res)
        else:
            report.gerar_relatorio_completo(res)

    elif args.mode == "scenarios":
        cen = scenarios.simular_cenarios(df, iteracoes=args.iterations,
                                         investimento_por_startup=args.capital)
        df_cmp = scenarios.comparar(cen)
        out = report.OUTPUT_DIR / "cenarios_comparativo.csv"
        df_cmp.to_csv(out, index=False)
        print("\n── Comparativo de Cenários ──")
        print(df_cmp.to_string(index=False))
        print(f"\nSalvo: {out}")

    elif args.mode == "optimize":
        df_rank = optimizer.otimizar(df, n_startups_alvo=args.target,
                                     granularidade=args.grid,
                                     iteracoes=args.iterations,
                                     investimento_por_startup=args.capital)
        out = report.OUTPUT_DIR / "alocacao_otima.csv"
        df_rank.to_csv(out, index=False)
        print("\n── Top Alocações por Sortino ──")
        print(df_rank.to_string(index=False))
        print(f"\nSalvo: {out}")

    elif args.mode == "sensitivity":
        df_sens = sensitivity.analisar(df, iteracoes=args.iterations,
                                       investimento_por_startup=args.capital)
        out = report.OUTPUT_DIR / "sensibilidade.csv"
        df_sens.to_csv(out, index=False)
        print("\n── Sensibilidade (top 8 premissas) ──")
        print(df_sens.head(8).to_string(index=False))
        print(f"\nSalvo: {out}")

    elif args.mode == "full":
        # Cenário base
        res_base = monte_carlo.simular(df, iteracoes=args.iterations,
                                       investimento_por_startup=args.capital)
        report.gerar_relatorio_completo(res_base)

        # Cenários macro
        cen = scenarios.simular_cenarios(df, iteracoes=args.iterations,
                                         investimento_por_startup=args.capital)
        df_cmp = scenarios.comparar(cen)
        df_cmp.to_csv(report.OUTPUT_DIR / "cenarios_comparativo.csv", index=False)

        # Otimização
        df_rank = optimizer.otimizar(df, n_startups_alvo=args.target,
                                     granularidade=args.grid,
                                     iteracoes=max(args.iterations // 4, 2000),
                                     investimento_por_startup=args.capital)
        df_rank.to_csv(report.OUTPUT_DIR / "alocacao_otima.csv", index=False)

        # Sensibilidade (iterações reduzidas — 24 simulações)
        df_sens = sensitivity.analisar(df,
                                       iteracoes=max(args.iterations // 4, 2000),
                                       investimento_por_startup=args.capital)
        df_sens.to_csv(report.OUTPUT_DIR / "sensibilidade.csv", index=False)

        # Tearsheet com decisão IC
        path = tearsheet.gerar(res_base, cen, df_rank, df_sens)
        print(f"\nTearsheet executivo: {path}")


if __name__ == "__main__":
    main()
