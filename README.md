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
