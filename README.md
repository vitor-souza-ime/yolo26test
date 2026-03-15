# YOLO26 Comparison – Benchmark de Variantes YOLO em Vídeo

Este projeto apresenta um **benchmark comparativo entre diferentes variantes do modelo YOLO26** utilizando vídeos de teste. O objetivo é analisar o desempenho de cada versão da arquitetura em termos de:

* número de classes detectadas
* total de detecções realizadas
* tempo médio de inferência por frame

O script foi projetado para rodar facilmente em **Google Colab, Jupyter Notebook ou ambiente local**, gerando automaticamente **relatórios e gráficos comparativos**.

---

# Objetivo

Avaliar o desempenho das variantes do modelo YOLO26:

* YOLO26 Nano
* YOLO26 Small
* YOLO26 Medium
* YOLO26 Large
* YOLO26 XLarge

A comparação é feita utilizando:

* vídeos de teste
* detecção de objetos em múltiplos frames
* métricas quantitativas de desempenho

---

# Principais Funcionalidades

O script executa automaticamente:

1. Instalação das dependências necessárias
2. Download automático de vídeos de teste
3. Extração de frames do vídeo
4. Execução das variantes do YOLO
5. Coleta de métricas de desempenho
6. Geração de relatório textual
7. Geração de gráficos comparativos

---

# Variantes Avaliadas

O benchmark compara as seguintes versões do modelo:

| Variante | Descrição                                      |
| -------- | ---------------------------------------------- |
| YOLO26n  | Nano – modelo leve e rápido                    |
| YOLO26s  | Small – equilíbrio entre velocidade e precisão |
| YOLO26m  | Medium – maior capacidade de detecção          |
| YOLO26l  | Large – alta precisão                          |
| YOLO26x  | XLarge – máxima capacidade computacional       |

---

# Métricas Avaliadas

Durante o processamento do vídeo, o script calcula:

**1. Classes únicas detectadas**

Quantidade de diferentes objetos identificados.

**2. Total de detecções**

Número total de objetos detectados ao longo de todos os frames.

**3. Tempo médio por frame**

Velocidade de inferência do modelo.

---

# Estrutura do Experimento

O pipeline executado pelo script segue estas etapas:

1. Seleção do vídeo de teste
2. Download automático (se necessário)
3. Extração de frames
4. Execução de cada variante YOLO
5. Coleta de resultados
6. Geração de relatório
7. Geração de gráficos

---

# Dataset de Vídeos

O script possui um catálogo interno com diferentes vídeos de teste.

Exemplos:

* rua com pessoas e bicicletas
* corredor de supermercado
* mesa com frutas e legumes
* sala de aula
* tráfego de veículos
* zona de trabalho industrial

Também é possível utilizar:

* um vídeo local
* um vídeo via URL

---

# Dependências

As principais bibliotecas utilizadas são:

* ultralytics
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* NetworkX

---

# Instalação

Execute no Colab ou Jupyter:

```bash
pip install ultralytics opencv-python-headless
```

Caso execute localmente, também pode ser necessário instalar:

```bash
pip install torch matplotlib numpy
```

---

# Execução

Basta rodar o script Python:

```bash
python yolo26_comparison.py
```

ou executar a célula no **Google Colab / Jupyter Notebook**.

O script executará automaticamente todas as etapas.

---

# Configurações Principais

Antes de rodar o experimento, é possível ajustar algumas variáveis:

```python
VIDEO_ID = 1
FRAMES = 60
CONF = 0.8
IMGSZ = 640
DEVICE = None
```

Descrição:

| Parâmetro | Função                         |
| --------- | ------------------------------ |
| VIDEO_ID  | vídeo do catálogo interno      |
| FRAMES    | número de frames analisados    |
| CONF      | confiança mínima das detecções |
| IMGSZ     | resolução de entrada           |
| DEVICE    | cpu ou gpu                     |

---

# Uso de GPU

O script detecta automaticamente se há GPU disponível.

Se CUDA estiver disponível:

```
GPU detectada → usando CUDA
```

Caso contrário:

```
CUDA não encontrado → usando CPU
```

---

# Arquivos Gerados

Após a execução, dois arquivos são produzidos:

**Relatório textual**

```
yolo26_report.txt
```

Contém:

* resultados por modelo
* classes detectadas
* top detecções
* tempos de inferência

---

**Gráfico comparativo**

```
yolo26_comparison.png
```

Mostra:

* classes únicas detectadas
* total de detecções
* tempo médio por frame

---

# Exemplo de Saída

Durante a execução, o terminal exibirá:

```
YOLO26 | device=CUDA | 60 frames

YOLO26N
Classes únicas: 12
Total detecções: 210
Tempo/frame: 8.5 ms
```

---

# Visualização Gerada

O gráfico final contém três análises:

1. Classes únicas por modelo
2. Total de detecções
3. Velocidade de inferência

Isso permite comparar **precisão vs desempenho computacional** entre as variantes.

---

# Possíveis Extensões

Este projeto pode ser expandido para:

* benchmarking com datasets maiores
* avaliação de precisão (mAP)
* comparação com YOLOv8 ou YOLOv9
* análise de consumo de GPU
* detecção em tempo real com webcam

---

# Aplicações

Esse tipo de benchmark é útil em:

* pesquisa em visão computacional
* seleção de modelos para edge computing
* sistemas de vigilância
* veículos autônomos
* robótica

---

# Autor

Vitor Amadeu Souza

Áreas de interesse:

* Inteligência Artificial
* Visão Computacional
* Sistemas Embarcados
* Aprendizado de Máquina

---

# Licença

Este projeto é disponibilizado para fins **educacionais e acadêmicos**.
