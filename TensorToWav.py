
channels = 2  # 1 = mono 2 = stereo
length = 200
wavTensor = tf.zeros(shape=[length,channels], dtype="float32")
sampleRateTensor = 44.1

wavString = tf.audio.encode_wav(wavTensor, sampleRateTensor)

# waveString is a string suitable to be saved out to create a .wav audio file
