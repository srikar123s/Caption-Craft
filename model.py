import torch
import torch.nn as nn

class LSTMSpeechRecognition(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, output_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, input_lengths):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_dim, seq_len)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        return self.fc(lstm_out)
