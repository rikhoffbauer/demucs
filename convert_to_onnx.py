from demucs.htdemucs import HTDemucs
import torch.onnx
import math
from demucs.hdemucs import pad1d
from demucs.spec import spectro
from demucs.states import load_model

def _spec(x, hop_length, nfft):
        hl = hop_length
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z

audio_len = 44100
mix = torch.randn(1, 2, audio_len * 10) # simulating 10 seconds of audio

## dummy spectrogram relative 
spec = _spec(mix, 4096 // 4, 4096) # precompute spectrogram

#model = pretrained.get_model('htdemucs') # to grab the model url
url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th" # HTDemucs

th = torch.hub.load_state_dict_from_url(
            url, check_hash=True) # type: ignore

assert th['klass'] == HTDemucs

model = load_model(th)

model.eval()

torch.onnx.export(model, 
                (mix, spec), 
                "htdemucs.onnx", 
                export_params=True,
                opset_version=20,
                dynamo=True,
                verify=True,
                report=True,
            ).save("htdemucs.onnx")