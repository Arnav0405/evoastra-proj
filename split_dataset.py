import os
import json
import random
import shutil
import zipfile


source_dir = 'MSCOCO_test2014_broken_up\dataset_part_3'
zip_filename = 'dataset_part_3.zip'

if os.path.exists(source_dir):
    print(f"Creating zip file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files and subdirectories
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Create archive name (relative path from source directory)
                arcname = os.path.relpath(file_path, start=os.path.dirname(source_dir))
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    # Get zip file size
    zip_size = os.path.getsize(zip_filename)
    zip_size_mb = zip_size / (1024 * 1024)
    
    print(f"\nZip file created successfully!")
    print(f"File: {zip_filename}")
    print(f"Size: {zip_size_mb:.2f} MB")
    
else:
    print(f"Directory '{source_dir}' not found!")

# Load the captions_train2014.json file
with open('dataset_part_1\\annotations\\captions_part_1.json', 'r') as f:
    captions_data = json.load(f)

captioned_data = captions_data['images']
length = len(captioned_data)
print(f"Loaded captions data with {length} entries")
print(f"How many times can we divide {length} by 5? {length / 7}")

# Count files in train2014/train2014 directory
train_dir = 'dataset_part_1\\images'
if os.path.exists(train_dir):
    train_files = os.listdir(train_dir)
    # Filter only files (not directories)
    train_files = [f for f in train_files if os.path.isfile(os.path.join(train_dir, f))]
    print(f"Number of files in {train_dir}: {len(train_files)}")
else:
    print(f"Directory {train_dir} does not exist")

# Split remaining images into 10 parts

# # Calculate images per part
# images_per_part = 8278
# total_parts = 10

# print(f"\nSplitting {len(captioned_data)} images into {total_parts} parts of {images_per_part} images each")

# # Shuffle the data to ensure random distribution
# random.shuffle(captioned_data)

# # Create directories and split the data
# for part_num in range(total_parts):
#     # Calculate start and end indices for this part
#     start_idx = part_num * images_per_part
#     end_idx = start_idx + images_per_part
    
#     # Get images for this part
#     part_images = captioned_data[start_idx:end_idx]
    
#     # Create directory for this part
#     part_dir = f'dataset_part_{part_num + 1}'
#     images_dir = os.path.join(part_dir, 'images')
#     annotations_dir = os.path.join(part_dir, 'annotations')
    
#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(annotations_dir, exist_ok=True)
    
#     print(f"\nProcessing Part {part_num + 1}: {len(part_images)} images")
    
#     # Copy image files to the part directory
#     copied_count = 0
#     for img in part_images:
#         filename = img['file_name']
#         source_path = os.path.join(train_dir, filename)
#         dest_path = os.path.join(images_dir, filename)
        
#         if os.path.exists(source_path):
#             shutil.copy2(source_path, dest_path)
#             copied_count += 1
#         else:
#             print(f"  Warning: Image not found: {filename}")
    
#     print(f"  Copied {copied_count} images to {images_dir}")
    
#     # Create annotations JSON for this part
#     part_captions_data = captions_data.copy()
#     part_captions_data['images'] = part_images
    
#     # Filter annotations to only include those for images in this part
#     part_image_ids = {img['id'] for img in part_images}
#     part_captions_data['annotations'] = [
#         ann for ann in captions_data['annotations'] 
#         if ann['image_id'] in part_image_ids
#     ]
    
#     # Save the JSON file for this part
#     json_filename = f'captions_part_{part_num + 1}.json'
#     json_path = os.path.join(annotations_dir, json_filename)
    
#     with open(json_path, 'w') as f:
#         json.dump(part_captions_data, f, indent=2)
    
#     print(f"  Created annotation file: {json_path}")
#     print(f"  Images: {len(part_images)}, Annotations: {len(part_captions_data['annotations'])}")

# print("\nDataset splitting completed!")
# print("Created directories:")
# for i in range(total_parts):
#     print(f"  - dataset_part_{i + 1}/")
#     print(f"    - images/ ({images_per_part} images)")
#     print(f"    - annotations/ (captions_part_{i + 1}.json)")