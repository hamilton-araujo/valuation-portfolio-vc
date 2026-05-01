# TODO — Valuation e Simulação de Portfólio VC

## Fase 1 — Ingestão e Limpeza de Dados
- [ ] Criar `src/ingest.py` — download via Kaggle API + cache local em parquet
- [ ] Criar `src/cleaner.py` — downcasting de tipos numéricos para reduzir RAM
- [ ] Filtrar rodadas Seed até Series C
- [ ] Remoção de outliers via IQR (multiplicador 1.5) sobre distribuições log dos valores levantados

## Fase 2 — Imputação de Valuation (Ridge Regression)
- [ ] Criar `src/imputer.py` — separar base com/sem valuation pós-money
- [ ] Engenharia de features: capital levantado, estágio, setor, país, ano
- [ ] Treinar Ridge Regression segmentado por setor (Fintech, SaaS, etc.)
- [ ] Avaliar via K-Fold Cross-Validation (MAE e R²)
- [ ] Imputar valuations faltantes e reinserir no DataFrame principal

## Fase 3 — Simulação de Monte Carlo
- [ ] Criar `src/monte_carlo.py` — motor 100% vetorizado com NumPy
- [ ] Modelar risco de ruína via distribuição Binomial por estágio
- [ ] Modelar multiplicador de retorno via Log-Normal (scipy.stats.lognorm)
- [ ] Suportar 10.000 a 100.000 iterações configuráveis via CLI
- [ ] Calcular VaR 5% (pessimista) e VaR 95% (otimista)

## Fase 4 — Relatório e Visualizações
- [ ] Criar `src/report.py` — painel analítico no terminal (Valor Esperado, Mediana, VaR)
- [ ] Gerar histograma da distribuição de retornos do portfólio
- [ ] Gerar curva de distribuição acumulada (CDF) dos retornos
- [ ] Exportar resultados em CSV

## Fase 5 — CLI
- [ ] Criar `src/main.py` com argparse
- [ ] Parâmetros: `--iterations`, `--confidence`, `--stages`, `--sectors`, `--capital`
- [ ] Exibir painel completo no terminal ao final

## Fase 6 — Testes
- [ ] Testar IQR: outliers devem ser removidos corretamente
- [ ] Testar Ridge: R² > 0 em todos os setores
- [ ] Testar Monte Carlo: VaR 5% < Mediana < VaR 95%
- [ ] Testar CLI: parâmetros inválidos levantam erro legível

## Fase 7 — Upload (GitHub)
- [ ] Criar `.gitignore` (excluir data/, output/, venv/)
- [ ] `git init` e primeiro commit
- [ ] Criar repositório público no GitHub e fazer push
