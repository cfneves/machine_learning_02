# Classificação de Renda — Adult Income Dataset

### Portfólio de Machine Learning | Cláudio Ferreira Neves

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-red)
![Métrica](https://img.shields.io/badge/Métrica%20Principal-F1%20Score-green)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

---

## Sobre o projeto

Projeto de classificação binária para prever se uma pessoa ganha mais ou menos de 50 mil dólares por ano, usando dados socioeconômicos do Census Bureau americano (Adult Income Dataset).

Faz parte do portfólio de Machine Learning construído como **softskill para o mercado de trabalho**. O código é estruturado com didática e documentação detalhada, sendo usado também como material de referência em sala de aula.

| | |
|---|---|
| **Problema** | Classificação binária (<=50K / >50K) |
| **Dataset** | Adult Income — 45.225 instâncias, 15 atributos |
| **Métrica principal** | F1 Score |
| **Modelo final** | Gradient Boosting + RandomizedSearchCV |
| **XAI** | SHAP (global + local) |

---

## Pipeline

```
Dados brutos (train / validation / test)
        │
        ▼
   EDA ─────────── distribuições, correlações, desbalanceamento
        │
        ▼
   Pré-processamento
        ├── NANs     → imputação pela moda (calculada no treino)
        └── Outliers → capping pelo método IQR
        │
        ▼
   Feature Engineering
        ├── Label Encoding   → categorias para inteiros
        ├── StandardScaler   → z-score (média=0, desvio=1)
        └── SelectKBest      → top 12 features por Informação Mútua
        │
        ▼
   Modelagem ────── Logistic Regression → Decision Tree → Random Forest → Gradient Boosting
        │
        ▼
   Tuning ────────── RandomizedSearchCV (5-fold CV, 40 iterações, scoring=F1)
        │
        ▼
   XAI ────────────── SHAP: importância global + beeswarm + waterfall local
        │
        ▼
   Avaliação final ── test.csv (usado UMA única vez)
```

---

## Estrutura do repositório

```
.
├── modelo_renda.ipynb    # Notebook principal — 28 células documentadas
├── train.csv             # 70% — treino do modelo
├── validation.csv        # 15% — tuning e avaliação
├── test.csv              # 15% — avaliação final apenas (não usar para tuning)
├── desafio.pdf           # Especificação original do problema
├── CLAUDE.md             # Instruções do projeto para Claude Code
├── fig_*.png             # Gráficos gerados ao executar o notebook
└── README.md             # Este arquivo
```

---

## O dataset

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| age | Numérico | Idade da pessoa |
| workclass | Categórico | Tipo de empregador (Private, Gov, Self-emp...) |
| fnlwgt | Numérico | Peso amostral do Census Bureau |
| education | Categórico | Nível de escolaridade |
| educational-num | Numérico | Anos de escolaridade (versão numérica) |
| marital-status | Categórico | Estado civil |
| occupation | Categórico | Ocupação profissional |
| relationship | Categórico | Papel na família |
| race | Categórico | Raça autodeclarada |
| gender | Categórico | Sexo |
| capital-gain | Numérico | Ganho de capital no ano |
| capital-loss | Numérico | Perda de capital no ano |
| hours-per-week | Numérico | Horas trabalhadas por semana |
| native-country | Categórico | País de origem |
| **income** | **Target** | **<=50K ou >50K** |

Valores ausentes estão codificados como `?` nos CSVs originais — tratados na célula de imputação.

---

## Por que F1 Score e não acurácia?

O dataset tem ~75% dos casos com renda `<=50K`. Um modelo que chuta "<=50K" para todos teria 75% de acurácia sem aprender absolutamente nada. O F1 Score equilibra precisão e recall, penalizando exatamente esse comportamento.

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)

Precisão = TP / (TP + FP)  →  dos que previ >50K, quantos realmente eram?
Recall   = TP / (TP + FN)  →  dos que eram >50K, quantos o modelo encontrou?
```

---

## Evolução do tuning

| # | Etapa | Configuração | F1 (val) |
|---|-------|-------------|----------|
| 1 | Regressão Logística (baseline) | max_iter=500, padrão | ~0.67 |
| 2 | Gradient Boosting padrão | n_estimators=100, lr=0.1 | ~0.73 |
| 3 | GB + Feature Selection | top 12 features (MI) | ~0.73 |
| 4 | **GB + RandomizedSearchCV** | melhores hiperparâmetros | **~0.76+** |

*Execute o notebook para ver os valores exatos do seu ambiente.*

---

## Explicabilidade com SHAP

O projeto usa três visualizações SHAP:

| Gráfico | O que mostra |
|---------|-------------|
| Importância global | Features mais influentes em média em todo o dataset |
| Beeswarm | Como cada feature empurra a previsão para >50K ou <=50K, por observação |
| Waterfall | Explicação individual: por que o modelo decidiu X para uma pessoa específica |

---

## Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/cfneves/machine_learning_02.git
cd machine_learning_02

# 2. Abra o notebook
jupyter lab modelo_renda.ipynb
# ou no VS Code: abra o arquivo diretamente
```

A primeira célula instala automaticamente todas as dependências (`shap`, `lightgbm`, `xgboost`) via `%pip install` — basta executar em ordem.

---

## Conceitos abordados no notebook

**Estatística e dados**
- Distribuições, assimetria e detecção visual de outliers
- Correlação de Pearson vs Informação Mútua
- Desbalanceamento de classes e suas implicações

**Pré-processamento**
- Data leakage: por que fit só no treino
- Capping IQR vs remoção de outliers
- Label Encoding vs One-Hot Encoding

**Modelagem**
- Bias-variance tradeoff
- Ensemble: bagging (Random Forest) vs boosting (Gradient Boosting)
- Hiperparâmetros: o que cada um controla

**Avaliação**
- Precisão, Recall, F1, AUC-ROC — quando usar cada um
- Como ler uma matriz de confusão
- Por que o test set é sagrado

**XAI**
- SHAP values: fundamento em teoria dos jogos
- Explicações globais vs locais

---

## Referências

- [Adult Income Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)
- [SHAP — Documentação oficial](https://shap.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

## Autor

**Cláudio Ferreira Neves**

Portfólio de Machine Learning — construído como softskill para o mercado de trabalho.

[![GitHub](https://img.shields.io/badge/GitHub-cfneves-181717?logo=github)](https://github.com/cfneves)
