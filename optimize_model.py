import onnxscript

import onnx

# run model optimizer

model = onnx.load("htdemucs.onnx")

opt = onnxscript.optimizer.optimize(model)

onnx.save(opt, "htdemucs_optimized.onnx")