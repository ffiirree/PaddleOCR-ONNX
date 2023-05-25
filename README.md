# PaddleOCR-ONNX

`Natural Language` / `Latex` OCR models trained with `PaddleOCR` / `PyTorch` and deployed with ONNX Runtime Interface C++.

## OCR

- [ ] Natural Language
- [ ] LaTex Mathematical Formula

## Install

### Anaconda

Download from [Anaconda Distribution](https://www.anaconda.com/products/distribution) and install.

### PyTorch Environment

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

```shell
conda install -c conda-forge onnx
pip install onnxruntime     # cpu
pip install onnxruntime-gpu # gpu
```

### PaddleOCR Environment

### C++ ONNX Environment

#### ONNX

Download and decompress [ONNX](https://github.com/microsoft/onnxruntime/releases), rename the folder to 'onnxruntime' and move it to the project's root directory.

#### OpenCV

Ubuntn

```bash
sudo apt install libopencv-dev
```

Windows

Download opencv and put its path to the `Environment Path`.

#### Compilation

```bash
mkdir build && cd build
cmake ..
make -j10
```
