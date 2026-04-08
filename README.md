# Emotion Detection in a Game Engine

Versao local e modular do notebook utilizado em outra versão, preparada para:

- preprocessar o dataset em formato FER 
- balancear classes com augmentation
- recortar faces com MediaPipe
- treinar EfficientNet-B0
- avaliar checkpoint salvo
- rodar inferencia em imagem local

## Estrutura

- `src/emotion_local/data.py`: leitura, filtro, balanceamento e splits
- `src/emotion_local/dataset.py`: `Dataset` PyTorch e transforms
- `src/emotion_local/model.py`: EfficientNet-B0 e selecao de device
- `src/emotion_local/landmarks.py`: extracao e cache de landmarks do MediaPipe
- `src/emotion_local/training.py`: dataloaders, treino e avaliacao
- `src/emotion_local/inference.py`: predicao em imagem
- `src/emotion_local/cli.py`: ponto de entrada por linha de comando

## Instalar dependencias

```bash
pip install -r requirements.txt
```

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

Treinar com landmarks (atualmente com problemas no windows):

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

## Observacoes

- O notebook original dependia de Colab, `kagglehub` e upload manual. Essa versao usa arquivos locais.
- O recorte por face via MediaPipe pode reduzir throughput. Se quiser medir impacto, use `--disable-face-crop`.
- O pretreino da EfficientNet tenta usar pesos ImageNet. Se o download falhar, o codigo cai para pesos aleatorios.
- Cada treino gera uma pasta propria dentro de `results/` com checkpoint, historico, graficos e matriz de confusao.

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
