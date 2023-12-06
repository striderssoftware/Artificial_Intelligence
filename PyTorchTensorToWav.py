#!pip install torch torchvision

import torch
import torchvision
import torchaudio
import io
import pathlib
import tensorflow as tf

print ("strider was here")

waveform_tensor = torch.tensor((), dtype=torch.float32)
waveform_tensor = waveform_tensor.new_zeros(1, 16000)

buffer = io.BytesIO()

#  Save Tensor to buffer
torchaudio.save(buffer, waveform_tensor, sample_rate, format="wav")

#  Save buffer to File
with open("testcreatewav.wav", "wb") as f:
    f.write(buffer.getbuffer())

print ("Endeeeee woooooooooooo")
