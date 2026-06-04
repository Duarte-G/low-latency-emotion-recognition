# Emotion Detection in a Game Engine

Sistema de **reconhecimento de emoções faciais em tempo real** que classifica 4 emoções
(**Angry, Happy, Sad, Neutral**) a partir de imagens de câmera e disponibiliza o resultado
para uma aplicação em **Unity** via servidor HTTP. O classificador é uma **EfficientNet-B0**
(transfer learning a partir da ImageNet), com detecção/recorte facial e extração opcional de
landmarks via **MediaPipe**. É a versão local e modular de um notebook anterior (que dependia
do Google Colab).

O projeto cobre todo o ciclo, do experimento à aplicação:

- preprocessar e padronizar dois datasets (**FER-2013** e **AffectNet**) para 4 classes
- balancear classes do FER-2013 com augmentation (apenas no treino)
- recortar faces com MediaPipe (com fallback OpenCV Haar Cascade)
- extrair e cachear landmarks faciais (vetor de 936 dimensões) — opcional
- treinar a EfficientNet-B0 (com fusão opcional imagem + landmarks)
- avaliar checkpoints e rodar um benchmark de 16 configurações
- rodar inferência em imagem, webcam ou via API HTTP para a Unity

## Arquitetura

Arquitetura do modelo (backbone EfficientNet-B0 + cabeça de classificação com fusão de landmarks):

<!-- Coloque aqui a imagem da arquitetura do modelo (a mesma usada no TCC). -->
![Arquitetura do modelo](docs/images/arquitetura_efficientnet.png)

Arquitetura do sistema (cliente Unity ↔ API Flask):

<!-- Coloque aqui o diagrama de fluxo cliente-servidor (o mesmo usado no TCC). -->
![Arquitetura do sistema](docs/images/arquitetura_sistema.jpg)


## Estrutura

- `src/emotion_local/config.py`: dataclasses de configuracao e mapeamento das 4 classes
- `src/emotion_local/data.py`: leitura, filtro, balanceamento e splits (FER-2013 e AffectNet)
- `src/emotion_local/dataset.py`: `Dataset` PyTorch e transforms
- `src/emotion_local/experiments.py`: montagem das configuracoes de experimento
- `src/emotion_local/face_detection.py`: deteccao e recorte facial (MediaPipe + fallback Haar)
- `src/emotion_local/landmarks.py`: extracao e cache de landmarks do MediaPipe
- `src/emotion_local/mediapipe_runtime.py`: compatibilidade do MediaPipe (API Tasks / modelo .task)
- `src/emotion_local/model.py`: EfficientNet-B0, fusao com landmarks e selecao de device
- `src/emotion_local/training.py`: dataloaders, treino, early stopping e avaliacao
- `src/emotion_local/results.py`: graficos, matriz de confusao e relatorios por execucao
- `src/emotion_local/comparison.py`: consolidacao comparativa de varias execucoes
- `src/emotion_local/inference.py`: predicao em imagem/quadro e modo webcam
- `src/emotion_local/server.py`: servidor HTTP (Flask) para a Unity
- `src/emotion_local/cli.py`: ponto de entrada por linha de comando

## Instalar dependencias

```bash
pip install -r requirements.txt
```

Se voce quiser executar como pacote com `python -m emotion_local ...`, instale tambem o projeto em modo editavel:

```bash
python3 -m pip install -e .
```

Sem essa instalacao, como o projeto usa layout `src/`, rode a CLI direto do codigo-fonte com `python3 -m src.emotion_local.cli ...` ou prefixe `PYTHONPATH=src`.

Para usar uma GPU, deve ser instalado uma build do PyTorch com CUDA. Se `torch.cuda.is_available()` retornar `False`, o treino vai cair para CPU.

## Exemplo de uso

Preparar os splits:

```bash
python -m src.emotion_local.cli prepare --fer-csv caminho/para/fer2013.csv --output-dir artifacts
```

Treinar:

```bash
python -m src.emotion_local.cli train --fer-csv caminho/para/fer2013.csv --output-dir artifacts --epochs 10 --batch-size 32 --num-workers 4 --device auto
```

Treinar com landmarks:

```bash
python -m src.emotion_local.cli train --fer-csv caminho/para/fer2013.csv --output-dir artifacts --results-dir results --epochs 10 --batch-size 32 --num-workers 4 --device auto --use-landmarks
```

Avaliar:

```bash
python -m src.emotion_local.cli evaluate --fer-csv caminho/para/fer2013.csv --output-dir artifacts --checkpoint artifacts/best_emotion_model.pt --device auto
```

Predizer imagem:

```bash
python -m src.emotion_local.cli predict --checkpoint artifacts/best_emotion_model.pt --image caminho/para/imagem.jpg --device auto
```

Predizer webcam:

```bash
python -m src.emotion_local.cli webcam --checkpoint artifacts/best_emotion_model.pt --device auto
```

## Servidor para Unity

O comando `serve` sobe uma API HTTP simples para integrar o modelo com a Unity. A ideia e:

- a Unity captura a imagem da camera
- envia essa imagem para o endpoint `POST /predict`
- o servidor roda a inferencia
- devolve emocao, confianca e probabilidades

Para iniciar o servidor:

```bash
python -m src.emotion_local.cli serve --checkpoint artifacts/best_emotion_model.pt --host 0.0.0.0 --port 5000 --device auto
```

Se voce estiver usando um checkpoint salvo dentro de `results/`, use o caminho completo:

```bash
python -m src.emotion_local.cli serve --checkpoint results/caminho/best_emotion_model.pt --host 0.0.0.0 --port 5000 --device auto
```

Endpoints principais:

- `GET /health`: informa se o servidor esta de pe e mostra device e labels carregadas
- `GET /emotion`: retorna a ultima emocao processada
- `POST /predict`: recebe uma imagem no campo `image` e retorna a predicao atual

Exemplo rapido de teste local:

```bash
curl http://127.0.0.1:5000/health
```

## Observacoes

- O notebook original dependia de Colab, `kagglehub` e upload manual. Essa versao usa arquivos locais.
- O recorte por face via MediaPipe pode reduzir throughput. Se quiser medir impacto, use `--disable-face-crop`.
- O pretreino da EfficientNet tenta usar pesos ImageNet. Se o download falhar, o codigo cai para pesos aleatorios.
- Cada treino gera uma pasta propria dentro de `results/` com checkpoint, historico, graficos e matriz de confusao.
- O projeto inclui o bundle `assets/mediapipe/face_landmarker.task` para a API nova do MediaPipe no Windows/Linux/WSL.

## Fluxo de Experimentos do TCC

O projeto agora suporta:

- FER-2013 baseline
- FER-2013 com face crop
- AffectNet baseline
- AffectNet com face crop
- landmarks opcionais do MediaPipe
- avaliacao no proprio dataset
- teste cruzado entre FER-2013 e AffectNet

Menu guiado:

```bash
python -m src.emotion_local.cli wizard
```

No menu, agora existe tambem a opcao de benchmark completo para rodar todas as combinacoes durante a noite.

Benchmark completo direto pela CLI:

```bash
python -m src.emotion_local.cli benchmark
```

Ou, se o projeto estiver instalado com `python3 -m pip install -e .`:

```bash
python3 -m emotion_local benchmark
```

Exemplos diretos:

```bash
python -m src.emotion_local.cli train --train-dataset fer2013 --test-mode same_dataset --disable-face-crop
python -m src.emotion_local.cli train --train-dataset fer2013 --test-mode same_dataset
python -m src.emotion_local.cli train --train-dataset affectnet --test-mode same_dataset --disable-face-crop
python -m src.emotion_local.cli train --train-dataset affectnet --test-mode same_dataset
python -m src.emotion_local.cli train --train-dataset fer2013 --test-mode cross_dataset --test-dataset affectnet
python -m src.emotion_local.cli train --train-dataset fer2013 --test-mode same_dataset --use-landmarks
```

Estrutura esperada dos datasets:

```text
dataset/
  FER-2013/
    fer2013.csv
  AffectNet/
    train/
    validation/
```

No AffectNet, `train/` e usado para treino mais validacao interna, enquanto `validation/` vira o teste final padrao.

## Comparacao de Resultados

Depois de executar dois ou mais treinamentos, voce pode consolidar os resultados automaticamente.

Comparar os dois treinos mais recentes:

```bash
python -m src.emotion_local.cli compare --latest 2 --name comparacao_inicial
```

Fluxo sugerido para benchmark noturno:

1. Rode `python -m src.emotion_local.cli wizard` e escolha `Rodar todas as combinacoes (overnight)`, ou execute `python -m src.emotion_local.cli benchmark`.
2. Deixe o processo terminar todos os experimentos de FER-2013 e AffectNet com baseline, face crop, landmarks e teste cruzado.
3. No dia seguinte, use `python -m src.emotion_local.cli compare --latest 16 --name tabela_completa` para gerar a tabela consolidada.

O benchmark tambem salva automaticamente:

- `results/benchmarks/<timestamp>_benchmark_summary.json`: resumo de todos os experimentos executados
- `results/comparisons/<timestamp>_benchmark/`: comparacao automatica entre as execucoes que terminaram com sucesso

Comparar execucoes especificas:

```bash
python -m src.emotion_local.cli compare --run-dir results\\20260406_220000_fer-2013_self_fer-2013_img-only_facecrop_img224_bs32_ep10_lr1e-04_auto --run-dir results\\20260406_221500_affectnet_self_affectnet_img-only_facecrop_img224_bs32_ep10_lr1e-04_auto --name fer_vs_affectnet
```

Arquivos gerados:

- `results/comparisons/<timestamp>_<nome>/comparison.csv`: tabela consolidada para abrir no Excel ou usar no TCC
- `results/comparisons/<timestamp>_<nome>/comparison.json`: dados completos em JSON
- `results/comparisons/<timestamp>_<nome>/summary.txt`: resumo rapido com ranking por `test_accuracy`

Cada linha da comparacao inclui, entre outros campos:

- dataset de treino e de teste
- modo de teste (`same_dataset` ou `cross_dataset`)
- uso de face crop
- uso de landmarks
- `val_accuracy`, `val_f1`, `test_accuracy`, `test_f1`
- metricas por emocao extraidas do `classification_report`

## Estado Atual Recomendado

Os exemplos acima mais antigos que usam `--fer-csv` isoladamente ficaram defasados. Para o estado atual do projeto, considere este fluxo como o correto:

- use `python -m src.emotion_local.cli wizard` para configurar o experimento pelo menu
- use `--train-dataset fer2013` para o FER-2013
- use `--train-dataset affectnet` para o AffectNet
- use `--test-mode same_dataset` para teste no proprio dataset
- use `--test-mode cross_dataset --test-dataset ...` para teste cruzado
- use `--use-landmarks` quando o MediaPipe FaceMesh/FaceLandmarker estiver disponivel no ambiente

Observacoes metodologicas importantes:

- o FER-2013 e lido a partir do CSV
- o AffectNet e lido a partir das imagens `.jpg` nas pastas
- o FER-2013 e balanceado por augmentation apenas no split de treino
- o AffectNet nao recebe esse mesmo balanceamento artificial
- a inferencia por webcam exige um checkpoint em arquivo `.pt` ou `.pth`
- a pasta `best_emotion_model/` sozinha nao pode ser passada diretamente para `--checkpoint`

Exemplos atualizados:

```bash
python -m src.emotion_local.cli wizard
python -m src.emotion_local.cli train --train-dataset fer2013 --test-mode same_dataset
python -m src.emotion_local.cli train --train-dataset affectnet --test-mode same_dataset
python -m src.emotion_local.cli train --train-dataset fer2013 --test-mode cross_dataset --test-dataset affectnet
python -m src.emotion_local.cli webcam --checkpoint results\\caminho\\best_emotion_model.pt --device auto
python -m src.emotion_local.cli serve --checkpoint results\\caminho\\best_emotion_model.pt --host 0.0.0.0 --port 5000
```

Servidor para Unity:

- `GET /health`: status do servidor
- `GET /emotion`: ultima emocao processada
- `POST /predict`: recebe uma imagem no campo `image` e retorna emocao, confianca e probabilidades
