import os
import numpy as np
from PIL import Image

def check_image_resizing_and_normalization(image_dir, expected_size=(299, 299)):
    total_images = 0
    resize_issues = []
    normalization_issues = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        try:
            with Image.open(img_path) as img:
                total_images += 1
                
                # Check resize
                if img.size != expected_size:
                    resize_issues.append((img_name, img.size))
                
                # Convert to array and normalize
                img_array = np.asarray(img).astype('float32') / 255.0
                
                # Check normalization range
                if img_array.min() < 0.0 or img_array.max() > 1.0:
                    normalization_issues.append(img_name)

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

    print(f"\n✅ Total images checked: {total_images}")
    
    if resize_issues:
        print(f"\n⚠️ Resize issues in {len(resize_issues)} images:")
        for name, size in resize_issues[:10]:
            print(f" - {name}: size = {size}")
    else:
        print("🎯 All images have correct size.")

    if normalization_issues:
        print(f"\n⚠️ Normalization issues in {len(normalization_issues)} images:")
        for name in normalization_issues[:10]:
            print(f" - {name}")
    else:
        print("🎯 All images are properly normalized in range [0, 1].")

# 👉 Set your actual image directory path here
image_dir = "resized_images"
check_image_resizing_and_normalization(image_dir)
