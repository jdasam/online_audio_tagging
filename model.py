import torch
import torch.nn as nn
import torchaudio


class SpecModel(nn.Module):
  def __init__(self, sr, n_fft, hop_length, n_mels):
    super().__init__()
    self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=False)
    self.db_converter = torchaudio.transforms.AmplitudeToDB()
  
  def forward(self, x):
    mel_spec = self.mel_converter(x)
    return self.db_converter(mel_spec)

class AudioModel(nn.Module):
  def __init__(self, sr, n_fft, hop_length, n_mels, hidden_size, num_output):
    super().__init__()
    self.sr = sr
    self.spec_converter = SpecModel(sr, n_fft, hop_length, n_mels)
    self.spec_norm = nn.BatchNorm1d(n_mels)
    self.conv_layer = nn.Sequential(
      nn.Conv1d(n_mels, out_channels=hidden_size//8, kernel_size=3),
      nn.BatchNorm1d(hidden_size//8),
      nn.MaxPool1d(2),
      nn.ReLU(),
      nn.Conv1d(hidden_size//8, out_channels=hidden_size//4, kernel_size=3),
      nn.BatchNorm1d(hidden_size//4),
      nn.MaxPool1d(2),
      nn.ReLU(),     
      nn.Conv1d(hidden_size//4, out_channels=hidden_size//2, kernel_size=3),
      nn.BatchNorm1d(hidden_size//2),
      nn.MaxPool1d(2),
      nn.ReLU(),
      nn.Conv1d(hidden_size//2, out_channels=hidden_size, kernel_size=3),
      nn.BatchNorm1d(hidden_size),
      nn.ReLU(),
    )
    self.final_layer = nn.Linear(hidden_size, num_output)

  def get_spec(self, x):
    '''
    Get result of self.spec_converter
    x (torch.Tensor): audio samples (num_batch_size X num_audio_samples)
    '''
    return self.spec_converter(x)
  
  def get_without_final_pooling(self, x, input_is_spec=False):
    if not input_is_spec:
      x = self.get_spec(x) # num_batch X num_mel_bins X num_time_bins
    spec = self.spec_norm(x)
    out = self.conv_layer(spec)
    out = self.final_layer(out.permute(0,2,1))
    out = torch.sigmoid(out)
    return out
  
  def forward(self, x):
    spec = self.get_spec(x) # num_batch X num_mel_bins X num_time_bins
    spec = self.spec_norm(spec)
    out = self.conv_layer(spec)
    out = torch.max(out, dim=-1)[0] # select [0] because torch.max outputs tuple of (value, index)
    out = self.final_layer(out)
    out = torch.sigmoid(out)
    return out

