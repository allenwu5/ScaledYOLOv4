# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

import numpy as np
import onnx
import onnxruntime
import torch.onnx

from models.models import Darknet, load_darknet_weights

batch_size = 1

width = 640
height = 384
device = 'cpu'
cfg = 'models/yolov4-csp.cfg'
weights = ['/ScaledYOLOv4/yolov4-csp.weights']
model = Darknet(cfg, width)
# model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
model = model.to(device)
load_darknet_weights(model, weights[0])

# set the model to inference mode
model.eval()

# Input to the model
x = torch.randn(batch_size, 3, height, width, requires_grad=True)
torch_out = model(x)

onnx_file_name = "yolov4_csp.onnx"

torch.onnx.export(model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  onnx_file_name,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable lenght axes
                                'output': {0: 'batch_size'}})

# Check model
onnx_model = onnx.load(onnx_file_name)
onnx.checker.check_model(onnx_model)


# Check output
ort_session = onnxruntime.InferenceSession(onnx_file_name)


def to_numpy(x):
    # return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    try:
        return x.detach().numpy()
    except:
        return x.numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(
    to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
