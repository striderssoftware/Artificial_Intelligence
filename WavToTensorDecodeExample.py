import pathlib
import tensorflow as tf

DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    tf.keras.utils.get_file(
    'mini_speech_commands.zip',
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = 'data/mini_speech_commands/yes/004ae714_nohash_0.wav'

print (data_dir)
file_contents = tf.io.read_file(data_dir)
contents, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=2)  #1= mono 2 = stereo

print (contents)

