print ("strider was here")
print(torch.__version__)

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

# open wav  as a tensor
waveform, sample_rate = torchaudio.load(data_dir)

print (waveform)
print (sample_rate)

print ("Endeeeee woooooooooooo")
