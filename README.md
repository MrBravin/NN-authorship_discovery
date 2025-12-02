# NN-authorship_discovery

Repositório de códigos e experimentos em redes neurais para **distinção de autoria** em manuscritos digitalizados e textos digitados de acervos brasileiros.

Este projeto reúne os scripts e notebooks usados em um Trabalho de Conclusão de Curso da Ilum – Escola de Ciência, que investiga métodos computacionais para comparação de autoria:

> **ENTRE A ESCRITA E O ESCRITO: ANÁLISE DE AUTORIA VIA PROCESSAMENTO DE IMAGEM E LINGUAGEM NATURAL**  
> Ilum – Escola de Ciência / CNPEM, 2025.
---

 Ele combina:
- **Processamento de Imagem (PdI)** – redes neurais siamesas convolucionais aplicadas a linhas de manuscritos digitalizados;
- **Processamento de Linguagem Natural (PLN)** – modelos baseados em BERTimbau e classificadores de machine learning para textos acadêmicos em português.

O objetivo central é **apoiar a perícia grafoscópica e a pesquisa histórica**, oferecendo pipelines reproduzíveis para comparar autoria em documentos de origem brasileira.

---

## Organização do repositório

A estrutura geral do projeto é dividida em dois eixos principais:

- `processamento_de_imagem/`  
  Scripts e notebooks para:
  - pré-processar páginas manuscritas (limpeza, escala de cinza, binarização opcional);
  - segmentar páginas em linhas de escrita;
  - montar pares de imagens por autor (autoria igual vs. autoria diferente);
  - definir, treinar e avaliar redes **siamesas CNN** (com backbone tipo TinyResNet);
  - calcular métricas de desempenho (por exemplo, ROC-AUC, F1, curvas ROC).

- `processamento_de_texto/`  
  Scripts e notebooks para:
  - extrair textos e metadados de PDFs (teses/dissertações);
  - realizar limpeza e normalização textual;
  - vetorização com **BERTimbau** (tokenização e geração de embeddings);
  - construção de representações de pares de autores (diferença, soma, concatenação de vetores);
  - treinamento e comparação de modelos (SVM, Random Forest, redes siamesas, etc.);
  - rotinas de busca de hiperparâmetros e de combinação de features.

Além desses diretórios principais, o repositório contem:

- pastas de **experimentos/resultados**, com gráficos, tabelas e logs de treino;
- arquivos de **modelos salvos** (`.pkl`, checkpoints, etc.);
- módulos utilitários com funções de apoio ao carregamento e pré-processamento de dados.

> Dica: abra os notebooks em cada pasta para ver o pipeline completo, do dado bruto até as métricas finais.

---

## Dados

Os pipelines foram pensados para trabalhar com dois tipos de conjuntos de dados:

- **Manuscritos digitalizados**  
  Páginas manuscritas de acervos históricos brasileiros, segmentadas em linhas para alimentar as redes siamesas de imagem.

- **Textos digitados**  
  Corpus de textos acadêmicos em português (por exemplo, teses do Domínio Público), usados para treinar modelos de PLN voltados à tarefa de autoria.

Por questões de **tamanho** e/ou **direitos autorais**, alguns conjuntos de dados podem não estar incluídos diretamente no repositório. Caso você queira reproduzir os experimentos:

1. obtenha seus próprios manuscritos digitalizados e textos em PDF/UTF-8;
2. organize-os nas pastas esperadas pelos notebooks (ajustando os caminhos conforme necessário);
3. mantenha a mesma lógica de estrutura: pastas por autor, listas de pares, divisões de treino/validação/teste, etc.

---

## Requisitos e ambiente

Recomenda-se:

- Python 3.10 ou superior;
- Abra e inicialise cada arquivo com calma resolvendo as dependencias de bibliotecas individualemente. Devido a grande variedade de arquivos ".ipynb", talvez mais de um ambiente virtual seja necessário. 

---

## Criadores:
As funções utilizadas para qualquer manipulação de dados dentro desse diretório assim como os métodos propostos foram criadas pelos estudantes: 
  - Pedro Henrique Kramer Canhim, email: pedro23013@ilum.cnpem.br.  
    Aluno do 6° Semestre do Curso de Bacharel em Ciência e Tecnologia, Ilum - Escola de Ciências.
  
  - João Pedro da Silva Mariano, email: joao23009@ilum.cnpem.br.  
    Aluna do 6° Semestre do Curso de Bacharel em Ciência e Tecnologia, Ilum - Escola de Ciências.

  Com o auxílio dos colaboradores:
  - Daniel Bravin Martins, email: daniel23020@ilum.cnpem.br
  - Diogo Pereira de Lima Carvalho, email: diogo23039@ilum.cnpem.br

O projeto passou pela orientação do pesquisador/doutor:
  - Amauri Jardim de Paula. Email: amauri.paula@ilum.cnpem.br
    Professor pesquisador da Ilum Escola de Ciências.

## Agradecimentos:
Agradecemos pela oportunidade e pelo suporte dos pesquisadores citados acima. Também agradecemos  agradecemos a todos aqueles que de alguma forma ajudaram na concepção de desse projeto.

---
