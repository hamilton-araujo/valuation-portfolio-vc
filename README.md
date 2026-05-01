# Portfólio VC — Decision Engine para Comitês de Investimento

> **A pergunta de US$ 50 milhões:** *Esse fundo de Venture Capital devolve 3× o capital em 10 anos? E se vier uma recessão? E qual a alocação ideal entre Seed, Series A, B e C?*

Este projeto é o motor quantitativo que responde essas três perguntas — não com opinião, com **probabilidade**. Ele transforma 50 mil rodadas de financiamento globais (Crunchbase / Kaggle) em uma **recomendação executiva**: APROVAR, REVISAR ou REJEITAR a tese de portfólio.

---

## Por que existe

Comitês de Investimento (IC) de fundos VC tomam decisões irreversíveis com base em três artefatos:

1. **Distribuição de retornos** — não a média, mas o intervalo: "qual a chance de eu perder dinheiro?"
2. **Stress-test macro** — "como ficamos se 2008 acontecer de novo?"
3. **Alocação ótima** — "qual mix de estágios maximiza retorno ajustado a risco?"

Modelos baseados em planilha Excel falham nas três frentes: ignoram a *power law* característica de VC, não simulam choques estruturais e tratam a alocação como decisão qualitativa. Este motor resolve as três num único pipeline.

---

## A história em três atos

### Ato 1 — O analista
Você é analista de um fundo VC de US$ 50M. Recebeu uma lista de 30 startups candidatas. O CIO quer uma recomendação até sexta. Você roda:

```bash
python src/main.py --mode full --iterations 50000
```

### Ato 2 — A decisão
O motor entrega um **tearsheet executivo em markdown** (`output/tearsheet_executivo.md`) com:

- Distribuição completa de retornos (Monte Carlo 50k iterações)
- Comparativo de 3 cenários macro (Base / Recessão 2008-like / Boom 2021-like)
- Top-10 alocações por estágio ranqueadas por **Sortino Ratio**
- Tornado de sensibilidade — quais 3 premissas movem 80% do resultado
- **Recomendação IC: APROVAR / REVISAR / REJEITAR** com justificativa quantitativa

### Ato 3 — A defesa
Quando o LP perguntar "e se a probabilidade de falha Seed for 80% em vez de 70%?", você abre o tornado chart e mostra: o múltiplo mediano cai de 2.4× para 1.9× — ainda dentro da zona aceitável.

---

## Stack

| Camada | Tecnologia | Por quê |
|---|---|---|
| Ingestão | Kaggle API · pandas (downcasting) | 50k registros em < 200 MB de RAM |
| Limpeza | IQR em escala log | preserva power law, descarta apenas outliers severos |
| Imputação | Ridge Regression segmentada por setor + K-Fold CV | corrige assimetria de informação no mercado privado |
| Simulação | NumPy vetorizado · scipy.stats.lognorm | 100k iterações em segundos, sem loops Python |
| Stress-test | Choques estruturais em P(falha) e parâmetros LogNormal | reproduz 2008 e 2021 |
| Otimização | Grid search 4D + Sortino Ratio | downside-only (LPs não punem upside) |
| Decisão | Heurística calibrada por benchmarks Cambridge Associates | mediana global VC ≈ 2.0× TVPI |

---

## Modelo

### Mortalidade (Binomial por estágio)

| Estágio | P(Falha) base |
|---|---|
| Seed | 70% |
| Series A | 50% |
| Series B | 35% |
| Series C | 25% |

### Multiplicador de retorno (LogNormal)

```
R ~ LogNormal(μ, σ)   amostrado por estágio
```

Captura a *power law*: minoria de ativos gera maioria do retorno.

### Cenários macro (choques estruturais)

| Cenário | Δ P(Falha) | Δ μ | Fator σ |
|---|---|---|---|
| Base | 0 | 0 | 1.00 |
| Recessão (2008-like) | +20pp | −0.4 | 0.70 |
| Boom (2021-like) | −15pp | +0.3 | 1.25 |

### Decisão IC (heurística)

| Decisão | Critério |
|---|---|
| ✅ APROVAR | P(3×) base ≥ 30% **E** P(perda) recessão ≤ 25% |
| ⚠️ REVISAR | qualquer um dos limites na zona intermediária |
| ❌ REJEITAR | P(3×) base < 15% **OU** P(perda) recessão > 40% |

---

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/ingest.py    # baixa via Kaggle API

# Modo único — Monte Carlo simples
python src/main.py --mode single --iterations 10000

# Comparativo de cenários macro
python src/main.py --mode scenarios --iterations 20000

# Otimização de alocação (grid 10%)
python src/main.py --mode optimize --target 30 --grid 0.10

# Sensibilidade (tornado das premissas)
python src/main.py --mode sensitivity --iterations 5000

# Tudo + tearsheet executivo
python src/main.py --mode full --iterations 20000
```

---

## Outputs

```
output/
├── tearsheet_executivo.md       # Recomendação IC + justificativa
├── cenarios_comparativo.png     # CDFs sobrepostas + métricas-chave
├── cenarios_comparativo.csv
├── sensibilidade_tornado.png    # Tornado das 8 premissas mais sensíveis
├── sensibilidade.csv
├── alocacao_otima.csv           # Top-10 alocações por Sortino
├── histograma_retornos.png
├── cdf_retornos.png
└── resultados_mc.csv            # Série completa de retornos simulados
```

---

## Estrutura

```
├── src/
│   ├── main.py           # CLI com 5 modos
│   ├── ingest.py         # Kaggle API + cache parquet
│   ├── cleaner.py        # IQR log, downcasting, feature engineering
│   ├── imputer.py        # Ridge por setor + K-Fold
│   ├── monte_carlo.py    # Simulação vetorizada (100k em segundos)
│   ├── scenarios.py      # Stress-test macro Base/Recessão/Boom
│   ├── optimizer.py      # Grid search + Sortino Ratio
│   ├── sensitivity.py    # Tornado ±15% ceteris paribus
│   ├── tearsheet.py      # Markdown executivo + decisão IC
│   └── report.py         # Painel CLI + gráficos
├── data/                 # raw + cache parquet (gitignored)
├── output/               # tearsheet, gráficos, CSVs
├── tests/
└── requirements.txt
```
