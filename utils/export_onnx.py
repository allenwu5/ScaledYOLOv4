import torch.onnx
import torchreid

# model_name = 'resnet18'
model_name = 'osnet_ain_x1_0'
epoch = 10

torchreid.models.show_avai_models()
extractor = torchreid.utils.FeatureExtractor(
    model_name=model_name,
    model_path=f'log/{model_name}/model/model.pth.tar-{epoch}',
    device=("cuda:0")
)
torch_model = extractor.model

batch_size = 1

# Input to the model
x = torch.randn(batch_size, 3, 256, 128, requires_grad=True).cuda()
torch_out = torch_model(x)

onnx_file_name = "person_reid_extractor.onnx"

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_file_name,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

# Check model
import onnx

onnx_model = onnx.load(onnx_file_name)
onnx.checker.check_model(onnx_model)


import numpy as np
# Check output
import onnxruntime

ort_session = onnxruntime.InferenceSession(onnx_file_name)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
