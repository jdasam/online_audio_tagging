from flask import Flask, render_template, jsonify
import pyaudio
from tagger import OnlineTagger, load_model
from mic_stream import MicrophoneStream
import numpy as np
from threading import Thread
import queue

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
print('http://127.0.0.1:5000/')
app = Flask(__name__)
global Q
Q = queue.Queue()
global model


@app.route('/')
def home():
    # args = Args()
    # model = load_model(args)
    global model
    model = load_model('audioset_model_large_gradual_epoch100.pt')
    global Q
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, Q))
    t1.start()
    return render_template('home.html')

@app.route('/_tag', methods= ['GET', 'POST'])
def tag():
    global Q
    probs = []
    while Q.qsize() > 0:
      results = Q.get()
      # tags.append(results['vocab'])
      probs.append(results['prob'])
    return jsonify(prob=probs)

@app.route('/_vocab', methods= ['GET', 'POST'])
def vocab():
  return jsonify(vocab=model.vocab.tolist())

def get_buffer_and_transcribe(model, q):
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
    RATE = 16000

    stream = MicrophoneStream(RATE, CHUNK, CHANNELS)
    tagger = OnlineTagger(model)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        stream.generator()
        print("* recording")
        while True:
          data = stream._buff.get()
          decoded = np.frombuffer(data, dtype=np.int16) / 32768
          if CHANNELS > 1:
              decoded = decoded.reshape(-1, CHANNELS)
              decoded = np.mean(decoded, axis=1)
          frame_output = tagger.inference(decoded, return_dict=True)
          q.put(frame_output)
          # print(sum(frame_output))
        stream.closed = True
    print("* done recording")

if __name__ == '__main__':
    # for i in range(0, p.get_device_count()):
    #     print(i, p.get_device_info_by_index(i)['name'])

    app.run(debug=True)
