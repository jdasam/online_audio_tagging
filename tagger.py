import torch
from model import AudioModel

class OnlineTagger:
  def __init__(self, model, threshold=0.8, recep_sample_size=19968):
    self.model = model
    self.hop_length = 512
    self.receptive_sample_size = recep_sample_size
    self.threshold = threshold
    self.vocab = self.model.vocab.tolist()

    self.audio_buffer = torch.zeros((1,self.receptive_sample_size))

  def update_buffer(self, audio):
    t_audio = torch.tensor(audio).to(torch.float)
    new_buffer = torch.zeros_like(self.audio_buffer)
    new_buffer[0, :-len(t_audio)] = self.audio_buffer[0, len(t_audio):]
    new_buffer[0, -len(t_audio):] = t_audio
    self.audio_buffer = new_buffer

  def inference(self, x, return_dict=False):
    with torch.no_grad():
      self.update_buffer(x)
      output = self.model(self.audio_buffer)
      output.squeeze_()
      if return_dict:
        return {'vocab': self.vocab, 'prob': output.tolist()}
      else:
        return self.model.vocab[output>self.threshold].tolist()



def load_model(pt_path):
  model = AudioModel(sr=16000, n_fft=1024, hop_length=512, n_mels=128, num_output=100, hidden_size=1024)
  saved_dict = torch.load(pt_path, map_location='cpu')
  model.load_state_dict(saved_dict['state'])
  model.vocab = saved_dict['vocab']
  model.eval()
  return model