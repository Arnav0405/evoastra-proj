import os
import shutil
import pandas as pd
def combine_images(source_folder_1, source_folder_2, dest_folder):
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Move images from both datasets
    for dataset_folder in [source_folder_1, source_folder_2]:
        for file_name in os.listdir(dataset_folder):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(dataset_folder, file_name), os.path.join(dest_folder, file_name))

# Paths to image folders for both datasets
flickr8k_images_path = 'Images'
flickr30k_images_path = 'flickr30k_images/flickr30k_images'
combined_images_path = 'Combined_Flickr_Images/Images'

# Combine images
combine_images(flickr8k_images_path, flickr30k_images_path, combined_images_path)
