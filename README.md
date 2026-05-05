# Classificação de Renda — Adult Income Dataset
### Portfólio | Machine Learning

---

## Sobre o projeto

Este projeto resolve um problema clássico de **classificação binária**: prever se uma pessoa ganha mais ou menos de 50 mil dólares por ano, usando dados socioeconômicos do Census Bureau americano.

Faz parte do meu portfólio de Machine Learning, construído para consolidar competências aplicadas ao mercado de trabalho. O código também é usado em sala de aula como material de referência para demonstrar um pipeline completo e bem estruturado.

**Métrica principal:** F1 Score  
**Dataset:** Adult Income (45.225 instâncias, 15 atributos)  
**Modelo final:** Gradient Boosting com RandomizedSearchCV

---

## Estrutura do repositório

```
.
├── modelo_renda.ipynb       # Notebook principal — pipeline completo
├── train.csv                # 70% dos dados — treino do modelo
├── validation.csv           # 15% dos dados — tuning e avaliação
├── test.csv                 # 15% dos dados — avaliação final apenas
├── desafio.pdf              # Especificação da atividade (SENAI)
├── fig_*.png                # Gráficos gerados pelo notebook
└── README.md                # Este arquivo
```

---

## O dataset

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| age | Numérico | Idade da pessoa |
| workclass | Categórico | Tipo de empregador (Private, Gov, etc.) |
| fnlwgt | Numérico | Peso amostral do Census |
| education | Categórico | Nível de escolaridade |
| educational-num | Numérico | Anos de estudo (versão numérica) |
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

Valores ausentes no original estão representados como `?` — tratados no notebook.

---

## Metodologia

### Por que F1 Score e não acurácia?

O dataset tem **~75% dos casos com renda <=50K**. Um modelo que prevê "<=50K" para todo mundo teria 75% de acurácia sem aprender nada. O F1 Score penaliza esse comportamento porque equilibra precisão (quando diz >50K, está certo?) e recall (quantos dos >50K ele captura?).

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```

### Pipeline completo

```
Dados brutos
    │
    ▼
EDA — entender distribuições, correlações, desbalanceamento
    │
    ▼
Pré-processamento
    ├── NANs: imputação pela moda (calculada no treino)
    └── Outliers: capping pelo método IQR
    │
    ▼
Feature Engineering
    ├── Label Encoding para variáveis categóricas
    ├── StandardScaler (z-score) para numéricas
    └── SelectKBest — Informação Mútua (top 12 features)
    │
    ▼
Modelagem — comparação de algoritmos
    ├── Regressão Logística (baseline)
    ├── Árvore de Decisão
    ├── Random Forest
    └── Gradient Boosting ← melhor
    │
    ▼
Tuning — RandomizedSearchCV (5-fold CV, 40 iterações)
    │
    ▼
XAI — SHAP Values (global + local)
    │
    ▼
Avaliação final no test.csv (usado UMA VEZ)
```

---

## Evolução do projeto (tabela de tuning)

| Etapa | Configuração principal | F1 Score (val) |
|-------|----------------------|----------------|
| 1 — Regressão Logística (baseline) | max_iter=500 | ~0.67 |
| 2 — Gradient Boosting padrão | n_estimators=100, lr=0.1 | ~0.73 |
| 3 — GB + Feature Selection | Top 12 features por MI | ~0.73 |
| 4 — **GB + RandomizedSearchCV** | Melhores hiperparâmetros | **~0.76+** |

*Valores exatos gerados ao executar o notebook.*

---

## Resultados finais (test set)

| Métrica | Resultado |
|---------|-----------|
| Accuracy | ver notebook |
| Precision | ver notebook |
| Recall | ver notebook |
| **F1 Score** | **ver notebook** |
| AUC-ROC | ver notebook |

*Execute o notebook para ver os valores precisos — eles variam conforme o ambiente.*

---

## Explicabilidade com SHAP

O projeto usa **SHAP (SHapley Additive exPlanations)** para responder:

- Quais features mais influenciam as previsões? (gráfico de importância global)
- Como cada feature empurra a previsão para >50K ou <=50K? (beeswarm plot)
- Por que o modelo previu X para *esta* pessoa específica? (waterfall plot)

Isso é fundamental em projetos reais — um modelo que ninguém consegue explicar raramente é aprovado para uso.

---

## Como executar

### Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

### Execução

1. Clone o repositório
2. Abra `modelo_renda.ipynb` no Jupyter Lab, Jupyter Notebook ou VS Code
3. Execute as células em ordem (Kernel → Run All)
4. Os gráficos são salvos automaticamente como `fig_*.png`

---

## Conceitos-chave para estudar após este projeto

**Pré-processamento**
- Por que a moda/mediana do treino — e não do dataset completo — deve ser usada na imputação?
- O que acontece se você aplicar o StandardScaler nos dados de teste separadamente?

**Modelagem**
- Qual a diferença entre Random Forest e Gradient Boosting?
- O que é overfitting e como a validação cruzada ajuda a detectá-lo?

**Métricas**
- Em que cenários Recall é mais importante que Precisão (e vice-versa)?
- AUC-ROC de 0.5 significa o quê?

**XAI**
- Por que SHAP é superior à importância de features nativa do sklearn?
- O que acontece quando duas features altamente correlacionadas estão no modelo?

---

## Referências

- [Adult Income Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)
- [SHAP — Documentação oficial](https://shap.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

---

## Autor

**Cláudio Ferreira Neves**

Projeto desenvolvido como parte do portfólio de Machine Learning — construído como softskill para o mercado de trabalho e publicado no GitHub.
