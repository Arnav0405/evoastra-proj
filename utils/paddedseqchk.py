import numpy as np

# Load the padded sequences
padded_captions = np.load('padded_captions.npy')

# Check shape and type
print("🧾 Shape:", padded_captions.shape)
print("📚 Data type:", padded_captions.dtype)

for i in range(10):
    print(f"Caption {i}:", padded_captions[i])
