

# Decode a 16-bit PCM WAV file to a float tensor.
file_contents = tf.io.read_file(filename)
contents, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)

    
