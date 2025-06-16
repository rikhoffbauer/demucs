# HTDemucs conversion to onnx

This repository contains the code to convert the HTDemucs model to onnx format. The model has been modified so that the stft and istft functions are placed outside the model, so now it expects the both the audio channels and the spectrogram as input.

`tools/export_onnx.py` can be used to convert any of the official models to
ONNX format. By default it will export all supported models:

```
python tools/export_onnx.py --all
```

Individual models can be exported with the `--model` flag.

The available models are `htdemucs`, `htdemucs_ft`, `htdemucs_6s`,
`hdemucs_mmi`, `mdx`, `mdx_extra`, `mdx_q` and `mdx_extra_q`.

A GitHub Actions workflow automatically runs this script for new releases and
uploads the resulting `.onnx` files as release assets.

branch `original` contains the original demucs code.

