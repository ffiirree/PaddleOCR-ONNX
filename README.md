# OCR_Torch2ONNX
Latex / English / Chinese OCR models trained with pytorch and deployed with ONNX

## OCR

- [ ] LaTex Mathematical Formula
- [ ] English
- [ ] Chinese

## Install

### Anaconda

Download from [Anaconda Distribution](https://www.anaconda.com/products/distribution) and install.

### Pytorch

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

```shell
conda install -c conda-forge onnx
pip install onnxruntime     # cpu
pip install onnxruntime-gpu # gpu
```