import os
from collections import defaultdict

# Your dataset folder structure
captions_file = "combined\combined_flickr8k_style_captions.txt"  # Original captions file from flickr
output_file = "cleaned_captions.txt"

# Dictionary to store cleaned captions
captions_dict = defaultdict(list)

# Read original captions and clean image names
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            image_name_with_id = parts[0]
            caption = parts[1].strip()

            # Remove the # and number (e.g., 1000268201_693b08cb0e.jpg#0 → 1000268201_693b08cb0e.jpg)
            image_name = image_name_with_id.split('#')[0]

            captions_dict[image_name].append(caption)

# Write cleaned captions to new file
with open(output_file, 'w', encoding='utf-8') as f:
    for image_name, captions in captions_dict.items():
        for caption in captions:
            f.write(f"{image_name}\t{caption}\n")

print(f"Cleaned and combined captions saved to: {output_file}")
