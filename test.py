import numpy as np

with open('data_extract_code\saved\decoder_input_data.npy', 'rb') as f:
    decoder_input_data = np.load(f)

print(decoder_input_data.shape)
print(decoder_input_data)