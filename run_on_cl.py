import pyaudio
from mic_stream import MicrophoneStream
from threading import Thread
import queue
import numpy as np
from tagger import OnlineTagger, load_model
import time

RATE = 16000
CHUNK = 4096
CHANNELS = 1


def get_buffer_and_transcribe(model, q):
  with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
      stream.generator()
      while True:
        data = stream._buff.get()
        decoded = np.frombuffer(data, dtype=np.int16) / 32768
        if CHANNELS > 1:
            decoded = decoded.reshape(CHANNELS, -1)
            decoded = np.mean(decoded, axis=0)
        frame_output = model.inference(decoded)
        q.put(frame_output)

def print_output(q):
  while True:
    updated_results = []
    while q.qsize():
      updated_results.append(q.get())
    for result in updated_results:
      print(result)
    time.sleep(0.250)

def main(model_file):
    model = OnlineTagger(load_model(model_file))

    q = queue.Queue()
    print("* recording")
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, q))
    t1.start()
    print('model is running')
    print_output(q)

    # print("* done recording")

if __name__ == '__main__':
    main('audioset_model_large_gradual2.pt')