import tensorflow as tf
import pdb
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

sess = tf.Session()

def load_wav_file(sess, filename):
  filename_ph = tf.placeholder(tf.string)
  loader = io_ops.read_file(filename_ph)
  decoder = contrib_audio.decode_wav(loader, desired_channels=1)
  return sess.run(decoder, feed_dict={filename_ph: filename})

def load_mfcc_file(sess, filename):
  filename_ph = tf.placeholder(tf.string)
  loader = io_ops.read_file(filename_ph)
  decoder = contrib_audio.decode_wav(loader, desired_channels=1, desired_samples=16000)
  spectrogram = contrib_audio.audio_spectrogram( decoder.audio, window_size=480, stride=160, magnitude_squared=True)
  mfcc = contrib_audio.mfcc( spectrogram, decoder.sample_rate, dct_coefficient_count=40)
  return sess.run(mfcc, feed_dict={filename_ph: filename})

def load_jpg_file(sess, filename):
  from PIL import Image
  jpgfile = Image.open(filename)
  pdb.set_trace()
  pdb.set_trace()

def wave_to_mfcc_jpeg(wave_filename, jpeg_filename):
  from python_speech_features import mfcc
  import scipy.io.wavfile as wav
  import matplotlib.pyplot as plt
  import numpy as np
  from matplotlib import cm

  (rate,sig) = wav.read("a.wav")
  mfcc_data = mfcc(sig,rate)

#def mfcc_to_jpeg(mfcc, jpeg_filename):
#  image = Image.new("RGB", 

#data = load_wav_file(sess, './data/spoken_numbers_wav/0/0_Fred_220.wav').audio
#data = load_mfcc_file(sess, './data/spoken_numbers_wav/0/0_Fred_220.wav')
pdb.set_trace()
pdb.set_trace()
load_jpg_file(sess, './a.jpg')

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

(rate,sig) = wav.read("a.wav")
mfcc_data = mfcc(sig,rate)

fig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC')
fig.savefig('./fig.jpeg')
plt.show()
