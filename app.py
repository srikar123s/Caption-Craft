import torch
import torchaudio
import torch.nn as nn

from torchaudio.transforms import MelSpectrogram

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import io
import soundfile as sf

from pathlib import Path


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# =========================================================
# YOUR EXISTING BILSTM MODEL
# =========================================================

class SpeechRecognitionModel(nn.Module):

    def __init__(
        self,
        input_dim=128,
        hidden_dim=256,
        output_dim=128
    ):

        super(SpeechRecognitionModel, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, x, input_lengths):

        x = x.transpose(1, 2)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(x_packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        out = self.fc(out)

        return out


# =========================================================
# LOAD YOUR MODEL
# =========================================================

model = SpeechRecognitionModel().to(device)


default_model_path = (
    Path(__file__).parent /
    "1nfine_tuned_model.pth"
)

# If your actual file has the original name,
# change the above line to:
#
# "1nfine_tuned_model.pth"


alt_model_path = Path(
    "C:/Users/srikar/OneDrive/Desktop/soft/"
    "1nfine_tuned_model(1).pth"
)


if default_model_path.exists():

    model_path = default_model_path

elif alt_model_path.exists():

    model_path = alt_model_path

else:

    raise FileNotFoundError(
        f"Model not found.\n"
        f"Expected: {default_model_path}"
    )


print(f"Loading your model from: {model_path}")


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("Your BiLSTM model loaded successfully.")


# =========================================================
# YOUR EXISTING PREDICTION FUNCTION
# =========================================================

def predict(waveform):

    mel_transform = MelSpectrogram(
        n_mels=128
    ).to(device)

    mel_spec = mel_transform(
        waveform
    ).squeeze(0)


    # Same 300-frame processing
    # as your original Caption Craft

    mel_spec = torch.cat(
        (
            mel_spec,
            torch.zeros(
                128,
                max(
                    0,
                    300 - mel_spec.shape[1]
                ),
                device=device
            )
        ),
        dim=1
    )[:, :300]


    mel_spec = mel_spec.unsqueeze(0).to(device)


    with torch.no_grad():

        input_length = torch.tensor(
            [mel_spec.shape[2]],
            device=device
        )

        output = model(
            mel_spec,
            input_length
        )

        pred = torch.argmax(
            output,
            dim=2
        )[0].cpu().numpy()


    chars = [
        chr(int(p))
        for p in pred
        if p != 0
    ]


    processed = []

    prev = ""

    for c in chars:

        if c != prev:

            processed.append(c)

        prev = c


    text = "".join(processed)

    text = (
        text
        .replace("~", "")
        .strip()
    )


    return text


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Caption Craft Hybrid Speech Recognition API",
    version="2.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# TRANSCRIBE ENDPOINT
# =========================================================

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...)
):

    if not audio:

        return {
            "error": "No audio uploaded."
        }


    try:

        audio_bytes = await audio.read()

        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_bytes)
        )


        # Stereo → mono

        if audio_data.ndim == 2:

            audio_data = audio_data.mean(
                axis=1
            )


        waveform = torch.tensor(
            audio_data
        ).float().unsqueeze(0).to(device)


        text = predict(waveform)


        return {
            "transcription": text
        }


    except Exception as e:

        return {
            "error": str(e)
        }