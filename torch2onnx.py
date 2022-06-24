import torch
import torchvision
import torchvision.transforms.functional as TransF
from PIL import Image
import torch.onnx

mobilenetv3s = torchvision.models.mobilenet_v3_small(pretrained=True).eval()

# water ouzel: target = 20
input = TransF.to_tensor(Image.open("water_ouzel.jpeg").convert('RGB'))
input = TransF.normalize(input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze_(0)

y = torch.softmax(mobilenetv3s(input), 1)

print('pytorch top5: ', torch.topk(y[0], 5, 0, True, True).indices)

# convert to onnx
torch.onnx.export(mobilenetv3s, input, 'mobilenetv3s.onnx', input_names=['input'], output_names=['output'])


# test onnx model
import onnx
import onnxruntime as ort

onnx_model = onnx.load('mobilenetv3s.onnx')
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession('mobilenetv3s.onnx')

# inference
y = ort_sess.run(None, { 'input' : input.numpy() })[0]

y = torch.softmax(TransF.to_tensor(y), 2)
print('onnx rt top5: ', torch.topk(y[0][0], 5, 0, True, True).indices)