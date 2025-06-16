import math
from argparse import ArgumentParser
from pathlib import Path

import torch

from demucs import pretrained
from demucs.hdemucs import pad1d
from demucs.spec import spectro

SUPPORTED_MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_extra",
    "mdx_q",
    "mdx_extra_q",
]


def _spec(x, hop_length, nfft):
    hl = hop_length
    le = int(math.ceil(x.shape[-1] / hl))
    pad = hl // 2 * 3
    x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
    z = spectro(x, nfft, hl)[..., :-1, :]
    assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
    return z[..., 2: 2 + le]


def export_model(name: str, out_dir: Path):
    model = pretrained.get_model(name)
    model.eval()
    mix = torch.randn(1, model.audio_channels, 10 * model.samplerate)
    spec = _spec(mix, model.hop_length, model.nfft)
    out_path = out_dir / f"{name}.onnx"
    export_output = torch.onnx.dynamo_export(
        model,
        mix,
        spec,
        opset_version=20,
        dynamic=True,  # set to True if variable-length input is required
    )
    export_output.save(out_path)
    print(f"Exported {name} to {out_path}")


def main():
    parser = ArgumentParser(description="Export Demucs models to ONNX")
    parser.add_argument(
        "--model",
        action="append",
        help="Model name to export. Can be used multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all supported models (default if no --model is given)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("onnx_models"),
        help="Directory where ONNX files will be written",
    )
    args = parser.parse_args()

    models = args.model or []
    if args.all or not models:
        models = SUPPORTED_MODELS

    args.out.mkdir(parents=True, exist_ok=True)
    for name in models:
        export_model(name, args.out)


if __name__ == "__main__":
    main()
