import torch
import torchaudio
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import soundfile as sf


# 🔹 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Define model
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, input_lengths):
        x = x.transpose(1, 2)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out)
        return out

# 🔹 Load model
model = SpeechRecognitionModel().to(device)
model.load_state_dict(torch.load("C:/Users/srikar/OneDrive/Desktop/soft/1nfine_tuned_model.pth", map_location=device))
model.eval()

# 🔹 Inference function
def predict(waveform):
    mel_transform = MelSpectrogram(n_mels=128).to(device)
    mel_spec = mel_transform(waveform).squeeze(0)
    mel_spec = torch.cat((mel_spec, torch.zeros(128, max(0, 300 - mel_spec.shape[1]), device=device)), dim=1)[:, :300]
    mel_spec = mel_spec.unsqueeze(0).to(device)

    with torch.no_grad():
        input_length = torch.tensor([mel_spec.shape[2]], device=device)
        output = model(mel_spec, input_length)
        pred = torch.argmax(output, dim=2)[0].cpu().numpy()

        chars = [chr(p) for p in pred if p != 0]
        processed = []
        prev = ""
        for c in chars:
            if c != prev:
                processed.append(c)
            prev = c

        return ''.join(processed).replace('~', '').strip()

# 🔹 FastAPI app
app = FastAPI()

# 🔹 Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not audio:
        return {"error": "No audio uploaded."}

    # 🔸 Read audio bytes
    audio_bytes = await audio.read()
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # 🔸 Convert to PyTorch tensor
    waveform = torch.tensor(audio_data).float().unsqueeze(0).to(device)

    # Run prediction
    text = predict(waveform)
    return {"transcription": text}

