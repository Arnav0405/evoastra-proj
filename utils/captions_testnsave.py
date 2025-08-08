import os
import csv

def verify_captions_with_images(captions_file, images_dir, output_csv):
    valid_data = []
    missing_images = set()

    with open(captions_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        if '\t' not in line:
            continue  # skip malformed lines
        image_name, caption = line.strip().split('\t')
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            valid_data.append((image_name, caption))
        else:
            missing_images.add(image_name)

    # Save valid entries to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'caption'])
        writer.writerows(valid_data)

    print(f"✅ Valid pairs saved to: {output_csv}")
    print(f"❌ Missing images: {len(missing_images)}")
    if missing_images:
        print("Example missing images:", list(missing_images)[:5])

# --- Usage ---
captions_file = 'cleaned_captions.txt'
images_dir = 'Images'
output_csv = 'valid_image_caption_pairs.csv'

verify_captions_with_images(captions_file, images_dir, output_csv)
