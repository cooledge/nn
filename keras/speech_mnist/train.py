import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Input
from keras.utils import to_categorical
from keras.optimizers import SGD
import argparse 
import input_data
import pdb

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='./data/spoken_numbers_wav',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectogram timeslices.',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      #default='yes,no,up,down,left,right,on,off,stop,go',
      default='0,1,2,3,4,5,6,7,8,9',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  FLAGS, unparsed = parser.parse_known_args()
  return FLAGS

FLAGS = setup_args()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#wav_data = load_wav_file('./data/spoken_numbers_wav/4_Daniel_220.wav')
#save_wav_file('./greg.wav', wav_data, 22050)

model_settings = prepare_model_settings(
    len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
    FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
    FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

audio_processor = input_data.AudioProcessor(
    '', FLAGS.data_dir, FLAGS.silence_percentage, 10,
    FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
    FLAGS.testing_percentage, model_settings)

audio_processor.prepare_background_data()

train_x, train_y = audio_processor.get_data(-1, 0, model_settings, 0.0, 0.0, 0, 'training', sess)
train_y = to_categorical(train_y, model_settings['label_count'])

#use_model = 'rnn'
use_model = 'cnn'
#use_model = 'simple'

print("Using model {0}".format(use_model))

input_frequency_size = model_settings['dct_coefficient_count']
input_time_size = model_settings['spectrogram_length']
def fixup_x(x):
  if use_model == 'simple':
    return x
  elif use_model == 'cnn':
    return x.reshape((-1, input_time_size, input_frequency_size, 1))
  elif use_model == 'rnn':
    return x.reshape((-1, input_time_size, input_frequency_size))

def simple_model():
  model = Sequential()
  model.add(Dense(256, input_dim=train_x.shape[1]))
  model.add(Activation('relu'))
  model.add(Dense(model_settings['label_count']))
  model.add(Activation('softmax'))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def cnn_model():
  model = Sequential()
  model.add(Conv2D(32, (3,3), input_shape=(input_time_size, input_frequency_size, 1)))
  model.add(Activation("relu"))

  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  #model.add(Dropout(0.9))

  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  #model.add(Dropout(0.8))

  model.add(Flatten())
  model.add(Dense(256))
  model.add(Dropout(0.8))
  model.add(Dense(model_settings['label_count']))
  model.add(Activation("softmax"))

  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def rnn_model():
  model = Sequential()
  model.add(LSTM(128, input_shape=(input_time_size, input_frequency_size)))
  model.add(Dropout(0.5))
  model.add(Dense(model_settings['label_count']))
  model.add(Activation("softmax"))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

train_x = fixup_x(train_x)
if use_model == 'simple':
  model = simple_model()
elif use_model == 'cnn':
  model = cnn_model()
elif use_model == 'rnn':
  model = rnn_model()
else:
  print("Unknown model, Blah")

model.fit(train_x, train_y, epochs=100, batch_size=32)

test_x, test_y = audio_processor.get_data(-1, 0, model_settings, 0.0, 0.0, 0, 'testing', sess)
test_y = to_categorical(test_y, model_settings['label_count'])
test_x = fixup_x(test_x)
score = model.evaluate(test_x, test_y)
names = model.metrics_names
print("{0}: {1}".format(names[1], score[1]))

