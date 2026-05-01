# Valuation e Simulação de Portfólio de Venture Capital

Modelo quantitativo para previsão de valuation e probabilidade de saída (IPO/Aquisição) de startups globais, com simulação estocástica de Monte Carlo sobre um portfólio massivo de Venture Capital.

## Dataset

**Startup Investments Dataset** (Kaggle) — +50.000 registros de rodadas de financiamento globais.

```bash
kaggle datasets download -d justinas/startup-investments
```

## Stack

| Camada | Tecnologia |
|---|---|
| Dados | Kaggle API · Pandas (downcasting otimizado) |
| Imputação | Scikit-learn · Ridge Regression · K-Fold CV |
| Simulação | NumPy · SciPy (lognorm) · Monte Carlo vetorizado |
| CLI | argparse |

## Como rodar

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Baixar dataset via Kaggle API
python src/ingest.py

# Rodar simulação com parâmetros padrão
python src/main.py --iterations 10000 --confidence 0.95

# Simulação mais robusta
python src/main.py --iterations 100000 --confidence 0.95 --stages seed series_a series_b
```

## Modelo de Risco

### Risco de Ruína (Mortalidade)
Cada startup recebe uma probabilidade de falha calibrada por estágio:

| Estágio | P(Falha) |
|---|---|
| Seed | 70% |
| Series A | 50% |
| Series B | 35% |
| Series C | 25% |

### Multiplicador de Retorno (Power Law)
Em caso de sobrevivência, o retorno segue uma distribuição **Log-Normal**:

```
R ~ LogNormal(μ, σ)   via scipy.stats.lognorm
```

Captura a assimetria característica de fundos VC: minoria de ativos gera maioria dos retornos.

### Imputação de Valuation (Ridge Regression)
```
Valuation ~ Ridge(α=L2) | features: [capital_levantado, estágio, setor, país, ano]
```
Treinado por setor via K-Fold Cross-Validation.

## Saídas do Painel CLI

```
══════════════════════════════════════════════
  SIMULAÇÃO MONTE CARLO — PORTFÓLIO VC
  Iterações: 10.000  |  Startups: 312
══════════════════════════════════════════════
  Valor Esperado (Média)    $  142.3 M
  Mediana                   $   98.7 M
  VaR 5%  (pessimista)      $   31.2 M
  VaR 95% (otimista)        $  410.8 M
  Probabilidade de 3x       34.2%
  Probabilidade de 10x      8.7%
══════════════════════════════════════════════
```

## Estrutura

```
├── src/
│   ├── main.py           # CLI principal
│   ├── ingest.py         # Download e cache do dataset Kaggle
│   ├── cleaner.py        # Limpeza, IQR, downcasting, filtro de estágios
│   ├── imputer.py        # Ridge Regression por setor + K-Fold CV
│   ├── monte_carlo.py    # Simulação vetorizada (sem loops Python)
│   └── report.py         # Painel analítico + gráficos
├── data/                 # Dataset bruto e processado (gitignored)
├── output/               # Gráficos e CSVs de resultado
├── requirements.txt
└── README.md
```
