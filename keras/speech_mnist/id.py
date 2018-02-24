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

#data = load_wav_file(sess, './data/spoken_numbers_wav/0/0_Fred_220.wav').audio
data = load_mfcc_file(sess, './data/spoken_numbers_wav/0/0_Fred_220.wav')
pdb.set_trace()
pdb.set_trace()
