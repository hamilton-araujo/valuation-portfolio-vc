"""
CLI principal — Valuation e Simulação de Portfólio de Venture Capital.

Exemplos:
    python src/main.py
    python src/main.py --iterations 50000 --confidence 0.95
    python src/main.py --stages Seed "Series A" --sectors Fintech SaaS
    python src/main.py --iterations 100000 --capital 500000 --no-report
"""

import argparse
import logging
import sys
from pathlib import Path

# Permite import direto de src/ quando rodado como script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest
import cleaner
import imputer
import monte_carlo
import report


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

    p.add_argument(
        "--iterations", type=int, default=10_000, metavar="N",
        help="Número de iterações Monte Carlo (padrão: 10000).",
    )
    p.add_argument(
        "--confidence", type=float, default=0.95, metavar="C",
        help="Nível de confiança para VaR (padrão: 0.95).",
    )
    p.add_argument(
        "--stages", nargs="+", default=None,
        metavar="ESTAGIO",
        help="Filtrar por estágios: Seed 'Series A' 'Series B' 'Series C'.",
    )
    p.add_argument(
        "--sectors", nargs="+", default=None,
        metavar="SETOR",
        help="Filtrar por setores (ex: Fintech SaaS Biotech).",
    )
    p.add_argument(
        "--capital", type=float, default=1_000_000, metavar="USD",
        help="Capital investido por startup em USD (padrão: 1000000).",
    )
    p.add_argument(
        "--iqr-multiplier", type=float, default=1.5, metavar="M",
        help="Multiplicador IQR para remoção de outliers (padrão: 1.5).",
    )
    p.add_argument(
        "--no-report", action="store_true",
        help="Suprimir geração de gráficos e CSV.",
    )
    p.add_argument(
        "--force-reload", action="store_true",
        help="Forçar reconstrução do cache parquet.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detalhado.",
    )

    return p.parse_args()


def _validar_args(args: argparse.Namespace) -> None:
    if args.iterations < 100:
        raise ValueError(f"--iterations deve ser ≥ 100 (recebido: {args.iterations}).")
    if not 0 < args.confidence < 1:
        raise ValueError(f"--confidence deve estar em (0, 1) (recebido: {args.confidence}).")
    if args.capital <= 0:
        raise ValueError(f"--capital deve ser positivo (recebido: {args.capital}).")
    if args.iqr_multiplier <= 0:
        raise ValueError(f"--iqr-multiplier deve ser positivo (recebido: {args.iqr_multiplier}).")


def main() -> None:
    args = _parse_args()
    _configurar_log(args.verbose)

    try:
        _validar_args(args)
    except ValueError as exc:
        print(f"Erro de parâmetro: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Ingestão ─────────────────────────────────────────────────────────────
    df = ingest.carregar(force_reload=args.force_reload)

    # ── Limpeza ──────────────────────────────────────────────────────────────
    df = cleaner.limpar(df, iqr_multiplier=args.iqr_multiplier)

    # ── Filtros opcionais ─────────────────────────────────────────────────────
    if args.stages:
        estagios_validos = {"Seed", "Series A", "Series B", "Series C"}
        invalidos = set(args.stages) - estagios_validos
        if invalidos:
            print(f"Estágios inválidos: {invalidos}. Válidos: {estagios_validos}", file=sys.stderr)
            sys.exit(1)
        df = df[df["estagio"].isin(args.stages)]

    if args.sectors:
        if "setor" in df.columns:
            df = df[df["setor"].isin(args.sectors)]
        else:
            print("Aviso: coluna 'setor' não encontrada — filtro de setores ignorado.", file=sys.stderr)

    if len(df) == 0:
        print("Nenhuma startup restante após filtros. Verifique --stages e --sectors.", file=sys.stderr)
        sys.exit(1)

    # ── Imputação ─────────────────────────────────────────────────────────────
    df = imputer.imputar(df)

    # ── Simulação Monte Carlo ─────────────────────────────────────────────────
    res = monte_carlo.simular(
        df,
        iteracoes=args.iterations,
        investimento_por_startup=args.capital,
    )

    # ── Relatório ─────────────────────────────────────────────────────────────
    if args.no_report:
        report.exibir_painel(res)
    else:
        report.gerar_relatorio_completo(res)


if __name__ == "__main__":
    main()
