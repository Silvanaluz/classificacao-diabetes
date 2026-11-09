# Deteccao de Diabetes - Classificacao com KNN, Arvore de Decisao e Rede Neural

Disciplina: Mineração de Dados
Alunas:Daniella Alves, 
Julia Ferreira e 
Silvana Luz

---

## Descricao do Projeto

Projeto de classificacao binaria para detectar diabetes a partir de dados tabulares de pacientes.
Sao implementados tres modelos:

- KNN (K-Vizinhos Mais Proximos)
- Arvore de Decisao
- Rede Neural (MLP - Perceptron Multicamadas)

Classes: Controle (0) e Diabetes (1).

---

## Requisitos

- Linux (testado no Fedora)
- Python 3.9 ou superior

---

## Instalacao



```
diabetes_classificacao/
├── kaggle-training set/
│   └── train.csv
├── kaggle-test/
│   └── test.csv
├── submission_template.csv
├── README.md
├── requirements.txt
├── src/
└── resultados/
```

os arquivos CSV estão dentro de pastas originais (kaggle-training set e kaggle-test).
O codigo ja esta configurado para encontra-los automaticamente nessas pastas.


```bash
cd diabetes_classificacao
python3 -m venv venv
source venv/bin/activate
```

### 3. foram  Instalados  dependencias

```bash
pip install -r requirements.txt
```

---

## Execucao

Executou-sepython src\07_gerar_submissao.py os scripts na ordem abaixo a partir da pasta raiz do projeto.
Todos os resultados sao salvos automaticamente na pasta resultados/.

### Analise Exploratoria

```bash
python3 src/01_exploracao.py
```

### Modelo KNN

```bash
python3 src/03_knn.py
```

### Modelo Arvore de Decisao

```bash
python3 src/04_arvore_decisao.py
```

### Modelo Rede Neural

```bash
python3 src/05_rede_neural.py
```

### Avaliacao Final e Comparacao dos Modelos

```bash
python3 src/06_avaliacao_final.py
```

O arquivo 02_preprocessamento.py e importado automaticamente pelos demais scripts
e nao precisa ser executado diretamente.

---

## Estrutura de Arquivos

```
diabetes_classificacao/
├── README.md                         <- este arquivo
├── requirements.txt                  <- dependencias Python
├── submission_template.csv           <- template de submissao Kaggle
├── kaggle-training set/
│   └── train.csv                     <- conjunto de treinamento
├── kaggle-test/
│   └── test.csv                      <- conjunto de teste
├── src/
│   ├── 01_exploracao.py              <- analise exploratoria dos dados
│   ├── 02_preprocessamento.py        <- limpeza e transformacao dos dados
│   ├── 03_knn.py                     <- modelo KNN
│   ├── 04_arvore_decisao.py          <- modelo arvore de decisao
│   ├── 05_rede_neural.py             <- modelo rede neural MLP
│   ├── 06_avaliacao_final.py         <- comparacao final dos tres modelos
│   └── preprocessamento_utils.py    <- utilitario de importacao
└── resultados/
    ├── 01_exploracao.txt
    ├── 03_knn_resultados.txt
    ├── 04_arvore_resultados.txt
    ├── 05_rede_neural_resultados.txt
    ├── 06_comparacao_modelos.txt
    └── 06_comparacao_modelos.csv
```

---

## Estrategia de Validacao

- O conjunto de TESTE nunca e usado durante o treinamento ou ajuste de hiperparametros.
- Um conjunto de VALIDACAO e criado a partir do treino (20% com estratificacao).
- A busca de hiperparametros usa validacao cruzada 5-fold no conjunto de treino.
- O conjunto de teste e usado APENAS na avaliacao final (script 06).

---

## Metricas Reportadas

Para cada modelo e conjunto (treino, validacao, teste):
- Acuracia
- Precisao
- Recall
- F1-Score
- Matriz de Confusao
