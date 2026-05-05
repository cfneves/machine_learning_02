---
name: ml-classificacao-renda
version: 1.0
description: Gerencia e evolui o projeto de classificação de renda (Adult Income Dataset). Atualiza modelos, métricas, tuning e documentação do notebook modelo_renda.ipynb.
allowed-tools: Read, Edit, NotebookEdit, Bash, Glob, Grep, Write
---

# Skill: ml-classificacao-renda

## Objetivo

Trabalhar com o projeto de classificação de renda (Adult Income Dataset).
O diretório de trabalho é aquele onde o repositório foi clonado.

Cobre: atualizar o notebook, ajustar pipeline, adicionar modelos, revisar tuning,
atualizar documentação e commitar para o GitHub.

## Contexto do projeto

- **Problema:** classificação binária — prever se renda é <=50K ou >50K
- **Dataset:** Adult Income Dataset (UCI) — 45.225 instâncias, 15 atributos
- **Métrica principal:** F1 Score (dataset desbalanceado ~75/25)
- **Modelo atual:** Gradient Boosting com RandomizedSearchCV
- **XAI:** SHAP (importância global + beeswarm + waterfall local)
- **Repositório:** https://github.com/cfneves/machine_learning_02
- **Autor:** Cláudio Ferreira Neves

## Arquivos principais

| Arquivo | Papel |
|---------|-------|
| `modelo_renda.ipynb` | Notebook principal — 28 células documentadas |
| `train.csv` | 70% — treino do modelo |
| `validation.csv` | 15% — tuning e avaliação |
| `test.csv` | 15% — avaliação final APENAS (nunca usar para tuning) |
| `CLAUDE.md` | Contexto completo do projeto para o Claude Code |
| `README.md` | Documentação pública do projeto |

## Regras obrigatórias

1. **Nunca usar `test.csv` para tuning** — apenas avaliação final
2. **Fit sempre no treino** — StandardScaler, LabelEncoder e SelectKBest ajustados só no train
3. **Moda e IQR calculados no treino** — aplicados em val/test sem recalcular
4. **Métrica de otimização:** sempre F1 Score (`scoring='f1'`)
5. **SHAP atualizado** sempre que o modelo final mudar
6. **Toda célula nova** deve ter cabeçalho `# === / # O que faz / # Resultado`
7. **Autor:** sempre Cláudio Ferreira Neves

## Mapa do notebook (células por ID)

```
02-imports        → instalação e imports
04-load           → carregamento dos CSVs
05/06/07          → EDA: head, info, describe
08/10/11/12       → EDA visual: target, numéricas, categóricas, correlação
14/15             → pré-proc: NaNs (análise + imputação)
17                → pré-proc: outliers (capping IQR)
19/20             → feature eng: separação X/y + Label Encoding
22                → feature eng: StandardScaler
24/25             → feature eng: MI + SelectKBest (K=12)
27/28             → modelagem: treino de 4 algoritmos + gráfico comparativo
30/31             → tuning: RandomizedSearchCV + tabela de evolução
33/34/35          → XAI: SHAP global, beeswarm, waterfall
37/38/39/40       → avaliação final: métricas, confusion matrix, relatório, resumo
```

## Como usar esta skill

### Adicionar um novo modelo

1. Abrir célula `27-baseline`
2. Inserir tupla na lista `modelos_base`:
   ```python
   ('Nome do Modelo', MinhaClasse(random_state=RANDOM_STATE))
   ```
3. Adicionar cor correspondente em `colors` na célula `28-plot-comparison`
4. Executar do `27-baseline` em diante
5. Se for o melhor modelo, atualizar `31-tuning-table` e as células SHAP/avaliação final

### Ajustar o tuning

1. Abrir célula `30-tuning`
2. Modificar `param_grid` (adicionar valores/chaves) ou `n_iter` (mais = melhor, mais lento)
3. Executar do `30-tuning` em diante
4. Registrar nova etapa em `31-tuning-table`

### Trocar o modelo final

1. Substituir `best_gb` pelo novo modelo nas células `33` a `40`
2. Garantir que o modelo implemente `predict_proba()` (necessário para AUC-ROC e SHAP TreeExplainer)
3. Atualizar README.md com os novos valores de F1

### Atualizar documentação

```bash
# Após qualquer mudança:
git add README.md CLAUDE.md modelo_renda.ipynb
git commit -m "tipo: descrição da mudança"
git push
```

## Output esperado ao usar esta skill

1. Ler o estado atual do notebook antes de qualquer modificação
2. Seguir a numeração/IDs de células existentes (não criar novos IDs sem necessidade)
3. Manter o padrão de cabeçalho em todas as células de código
4. Verificar ausência de data leakage em qualquer transformação adicionada
5. Atualizar README.md se resultados numéricos mudarem
6. Commitar com mensagem descritiva e fazer push ao final

## Instalação desta skill

```bash
# Windows (PowerShell)
$dest = "$env:USERPROFILE\.claude\Skills\ml-classificacao-renda"
New-Item -ItemType Directory -Force -Path $dest
Copy-Item "skills\ml-classificacao-renda.md" "$dest\SKILL.md"

# macOS / Linux
mkdir -p ~/.claude/Skills/ml-classificacao-renda
cp skills/ml-classificacao-renda.md ~/.claude/Skills/ml-classificacao-renda/SKILL.md
```
