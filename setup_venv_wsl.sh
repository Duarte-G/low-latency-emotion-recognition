#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3.11}
VENV_DIR="${1:-.venv}"

echo "==> Criando venv em: $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "==> Atualizando pip"
pip install --upgrade pip

echo "==> Instalando PyTorch com CUDA 12.8 (RTX 5070 Blackwell)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==> Instalando dependencias do projeto"
pip install \
    "numpy<2.3" \
    pandas \
    matplotlib \
    flask \
    opencv-python \
    scipy \
    scikit-learn \
    Pillow \
    mediapipe \
    tqdm \
    seaborn

echo "==> Instalando pacote local em modo editavel"
pip install -e "$(dirname "$0")"

echo ""
echo "Ambiente pronto. Para ativar:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Para treinar todas as combinacoes (benchmark completo):"
echo "  python -m emotion_local benchmark --train-dataset fer2013 --affectnet-dir /path/to/affectnet --fer-csv /path/to/fer2013.csv"
