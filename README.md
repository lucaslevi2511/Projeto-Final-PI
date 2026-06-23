# Classificação de Objetos Submarinos via CNN Customizada do Zero

Este repositório contém o código-fonte e o pipeline de processamento de imagens para o artigo focado na classificação de fauna e resíduos em ambientes aquáticos. O projeto lida com os desafios severos de qualidade de imagem encontrados no ambiente marinho através de uma abordagem que combina engenharia de recortes geométricos robustos e uma Rede Neural Convolucional (CNN) treinada inteiramente do zero.

---

## Descrição do Problema

A visão computacional em ambientes subaquáticos enfrenta obstáculos físicos severos que degradam a qualidade das imagens e dificultam o aprendizado de modelos de IA tradicionais. Este artigo busca mitigar problemas críticos como:

* **Atenuação da Luz e Perda de Cores:** Conforme a profundidade aumenta, as frequências de luz (especialmente o vermelho) são absorvidas, deixando as imagens predominantemente azuladas ou esverdeadas.
* **Forte Ruído Visual e Turbidez:** A presença de sedimentos em suspensão (neve marinha) cria partículas flutuantes que geram ruído de alta frequência e reduzem o contraste dos objetos.
* **Iluminação Desigual e Sombras:** A refração da luz solar na superfície da água cria padrões de iluminação dinâmicos e áreas de sombra densas.
* **Oclusão e Camuflagem:** Organismos como *Crab* (caranguejos) e *Coral-Reef* frequentemente se misturam ao fundo oceânico, exigindo que o modelo capture texturas finas e variações de borda para diferenciá-los.

---

## Técnicas Utilizadas

### 1.Pipeline de Pré-processamento e Recorte (Pipeline)
Para garantir a padronização geométrica e eliminar distorções estruturais antes da entrada na CNN, o fluxo inicial realiza as seguintes operações:

* **Dimensionamento Baseado em Anotações:** Conversão das coordenadas de 8 pontos do arquivo de mapeamento em caixas delimitadoras (bounding boxes) precisas.

* **Padding de Segurança:** Adição de uma margem fixa de 15 pixels ao redor do objeto para preservar os limites estruturais e evitar cortes abruptos nas bordas.

* **Centralização Quadrada (MAKE_SQUARE):** Expansão da matriz de recorte para um formato perfeitamente quadrado com preenchimento zero (zero-padding), prevenindo deformações morfológicas durante o redimensionamento.

* **Redimensionamento Canônico (Resize):** Redução da imagem para a resolução fixa de 224x224 pixels via interpolação por área (INTER_AREA), otimizando o tensor para a arquitetura de Deep Learning.

* **Alinhamento de Canais (BGR ➔ RGB):** Conversão do espaço de cor nativo do OpenCV para o padrão universal de tensores, garantindo a integridade cromática a ser interpretada pelo modelo.

### 2. Isolamento de Luminância e Suavização de Ruído (YCrCb & Bilateral)
Para tratar variações de iluminação e ruído sem corromper as informações originais de cor do objeto, a pipeline separa os componentes estruturais dos cromáticos:

* **Decomposição YCrCb:** Divisão da imagem nos canais de Luminância (Y) e Crominância (Cr, Cb), isolando o brilho e os gradientes texturais das matrizes puras de cor.

* **Filtragem Espacial Bilateral:** Aplicação do filtro bilateral exclusivamente no canal Y para atenuar ruídos digitais e granulados de compressão comuns em capturas subaquáticas.

* **Preservação de Bordas:** Utilização de métricas de distância espacial e variação de intensidade simultâneas, garantindo que superfícies homogêneas do fundo sejam suavizadas enquanto os contornos e texturas do animal permanecem intactos.

### 3. Realce de Contraste e Normalização Final (Equalization & Normalization)
A etapa final maximiza a visibilidade das características latentes e ajusta a escala numérica dos dados para a convergência da rede:

* **Mapeamento por CDF:** Cálculo do histograma de frequências do canal Y filtrado e computação da Função de Distribuição Acumulada para espalhar uniformemente os tons de intensidade.

* **Equalização Linear:** Interpolação matemática para esticar dinamicamente o contraste global da luminância no intervalo [0, 255], revelando microtexturas ocultas sob condições de baixa luminosidade.

* **Recomposição Cromática:** Fusão do canal Y equalizado com as matrizes originais Cr e Cb, seguida da reconversão direta e definitiva para o espaço RGB.

Escalonamento Flutuante (Normalizar): Divisão da matriz final por 255.0 para converter os valores de pixel para o intervalo decimal [0.0, 1.0], mitigando problemas de desvanecimento do gradiente durante as etapas de retropropagação da CNN.

### 2. Arquitetura da CNN (Do Zero)
Diferente de abordagens tradicionais que usam Transfer Learning (como ResNet ou VGG pré-treinadas em imagens terrestres do ImageNet), este trabalho propõe uma **CNN convolucional criada do zero**, forçando a extração de mapas de características puramente adaptados às frequências e cores do ambiente aquático.

A arquitetura segue o conceito de **Pirâmide de Abstração Visual**:
* **Blocos Convolucionais Crescentes:** Filtros que aumentam progressivamente (32 -> 64 -> 128 -> 256) para trocar resolução espacial por profundidade de conceitos (desde bordas simples até partes complexas dos objetos).
* **Normalização por Lote (`BatchNorm2d`):** Aplicada após cada convolução para estabilizar a escala dos gradientes e acelerar a convergência.
* **Camada de Funil Decrescente:** O classificador achata as características extraídas, comprimindo os dados em uma camada densa de 512 neurônios (protegida por **Dropout de 50%** contra overfitting) antes de reduzir para as 7 classes finais.

O dataset é dividido nativamente em **70% Treino**, **10% Validação** e **20% Teste**, cobrindo as seguintes classes:
1. `Fish` (Peixe)
2. `Crab` (Caranguejo)
3. `Human` (Humano/Mergulhador)
4. `Trash` (Lixo/Plástico)
5. `Jellyfish` (Água-viva)
6. `Coral Reef` (Recife de Coral)
7. `Fish Group` (Cardume)

---

## Como Rodar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado e instale as dependências exigidas executando:

```bash
pip install torch torchvision scikit-learn opencv-python numpy
```

Modos de Execução:
o repositório está organizado com dois scripts de entrada principais, dependendo do seu fluxo de trabalho:

Opção A: Executar Pipeline Completo (Processamento + Treino).
Se você quer, além de recortar processar as imagens com as operações listadas antes de iniciar o treinamento e a validação da CNN, utilize o arquivo main.py:

```bash
python main.py
```

O que ele faz: Cria a pasta saida_recortes/, lê os arquivos de imagens e as coordenadas dos arquivos .txt, recorta os alvos aplicando as margens quadradas, organiza-os em subpastas por classe e, imediatamente após, inicia o treinamento e exibe o relatório de métricas (Precision, Recall, F1-Score) ao final.

Opção B: Executar Apenas o Recorte e Treino da CNN.
Se o seu objetivo no momento é apenas recortar o dataset, e treinar a CNN sem mais operações, execute o arquivo mainsempipeline.py:

```bash
python mainsempipeline.py
```
O que ele faz: Executa exclusivamente a varredura das pastas train, valid e test originais, gera todos os recortes, salva tudo de forma estruturada e manda diretamente para treino da CNN, sem processamento.

Métricas Avaliadas
Ao final do treinamento realizado pelo script main.py, o console exibirá de forma detalhada:

Loss e Acurácia por Época (tanto para o conjunto de Treino quanto para o de Validação).

Matriz de Confusão Completa, permitindo identificar quais classes submarinas estão se confundindo (ex: se Fish Group está sendo classificado como Fish individual devido à turbidez).

Relatório de Classificação contendo os valores exatos de Precision, Recall (Sensibilidade para não deixar lixos ou espécimes passarem batidos) e o F1-Score por categoria.
