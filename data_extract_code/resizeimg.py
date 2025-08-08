from PIL import Image
import os

def resize_images(input_folder, output_folder=None, size=(299, 299)):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = input_folder  # overwrite in place

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        
        # Skip non-image files
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            save_path = os.path.join(output_folder, img_name)
            img.save(save_path)
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

    print(f"All images resized to {size} and saved in: {output_folder}")

# Example usage:
resize_images("Images", output_folder="resized_images")  # change folder path as needed
