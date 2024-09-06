# HTDemucs conversion to onnx

This repository contains the code to convert the HTDemucs model to onnx format. The model has been modified so that the stft and istft functions are placed outside the model, so now it expects the both the audio channels and the spectrogram as input.

Script `convert_to_onnx.py` is used to convert the model to onnx format.

branch `original` contains the original demucs code.

