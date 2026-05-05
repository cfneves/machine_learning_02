# CLAUDE.md — Classificação de Renda (Adult Income Dataset)

## O que é este projeto

Pipeline completo de Machine Learning para classificação binária de renda (<=50K vs >50K).
Portfólio de Cláudio Ferreira Neves — softskill para o mercado de trabalho.
Usado também como material de referência didática em aulas de Machine Learning.

## Arquivos principais

| Arquivo | Papel |
|---------|-------|
| `modelo_renda.ipynb` | Notebook principal — 28 células documentadas, executar em ordem |
| `train.csv` | 70% dos dados — usado para treinar modelos |
| `validation.csv` | 15% — usado para tuning e comparação de modelos |
| `test.csv` | 15% — **usado UMA VEZ, somente na avaliação final** |
| `desafio.pdf` | Especificação original do problema |

## Regras críticas do projeto

- **Nunca usar test.csv para tuning** — viola a regra de avaliação limpa (data leakage)
- **Fit sempre no treino** — StandardScaler, LabelEncoder e SelectKBest são ajustados no train e apenas transformam val/test
- **Moda calculada no treino** — imputação de NaNs usa estatísticas do train, aplicadas em val/test
- **Métrica principal é F1 Score** — acurácia é enganosa neste dataset (desbalanceado ~75/25)
- **Autor sempre: Cláudio Ferreira Neves** — portfólio profissional, não atividade acadêmica

## Stack e dependências

```
pandas, numpy, matplotlib, seaborn
scikit-learn (preprocessing, feature_selection, ensemble, metrics, model_selection)
shap          # XAI — instala via %pip na primeira célula
lightgbm      # instalado junto, disponível para experimentos futuros
xgboost       # instalado junto, disponível para experimentos futuros
```

## Estrutura do notebook (ordem das células)

```
Célula 1  — Instalação e imports
Célula 2  — Carregamento dos dados (na_values=' ?')
Células 3-5  — EDA: head, info, describe
Células 6-9  — EDA visual: target, numéricas, categóricas, correlação
Células 10-11 — Pré-proc: análise e imputação de NaNs
Célula 12  — Pré-proc: capping de outliers (IQR)
Células 13-14 — Feature Eng: separação X/y + Label Encoding
Célula 15  — Feature Eng: StandardScaler
Células 16-17 — Feature Eng: Informação Mútua + SelectKBest (K=12)
Células 18-19 — Modelagem: treino e comparação de 4 algoritmos
Células 20-21 — Tuning: RandomizedSearchCV + tabela de evolução
Células 22-24 — XAI: SHAP global, beeswarm, waterfall
Células 25-28 — Avaliação final: métricas, confusion matrix, relatório, resumo
```

## Como trabalhar neste projeto

### Adicionar um novo modelo
1. Instanciar o modelo na lista `modelos_base` (célula 18)
2. Adicionar uma cor em `colors` para o gráfico comparativo
3. Executar do célula 18 em diante

### Ajustar o tuning
- Alterar `n_iter` na célula 20 (mais iterações = melhor resultado, mais tempo)
- Expandir `param_grid` com novos hiperparâmetros ou valores

### Trocar o modelo final
- Substituir `best_gb` pelo novo modelo nas células 22-28
- Garantir que o modelo tenha `predict_proba()` (necessário para AUC-ROC e SHAP)

### Gráficos gerados
Todos os `fig_*.png` são criados automaticamente ao executar as células correspondentes.
Não commitar esses arquivos — estão no .gitignore (ou devem estar).

## Contexto de uso em aula

O notebook é usado para demonstrar:
- Por que fit/transform separam treino de val/test
- O impacto do desbalanceamento nas métricas
- A diferença prática entre os algoritmos de ensemble
- Como SHAP explica decisões de modelos caixa-preta

Manter linguagem técnica precisa e comentários didáticos em todas as células.
