import os
from collections import defaultdict

def combine_and_format_captions(
    captions_filepath: str,
    results_filepath: str,
    output_filepath: str
):
    """
    Combines captions from two input files (one CSV, one plain text)
    and formats them into the Flickr8k dataset caption file style.

    Args:
        captions_filepath (str): Path to the 'captions.txt' file (image,caption format).
        results_filepath (str): Path to the 'results.csv' file (image_name|comment_number|comment format).
        output_filepath (str): Path where the combined and formatted output file will be saved.
    """
    # Dictionary to store unique captions for each image
    # Key: image_name (e.g., '1000268201_693b08cb0e.jpg')
    # Value: set of captions (using a set to automatically handle duplicates)
    image_captions = defaultdict(set)

    print(f"Processing '{captions_filepath}'...")
    try:
        with open(captions_filepath, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline()
            if not header.strip().lower().startswith("image,caption"):
                print(f"Warning: '{captions_filepath}' does not appear to have the expected header 'image,caption'. Proceeding anyway.")

            for line_num, line in enumerate(f, 2): # Start line_num from 2 for data lines
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',', 1) # Split only on the first comma
                if len(parts) == 2:
                    image_name = parts[0].strip()
                    caption = parts[1].strip()
                    image_captions[image_name].add(caption)
                else:
                    print(f"Warning: Skipping malformed line {line_num} in '{captions_filepath}': '{line}'")
    except FileNotFoundError:
        print(f"Error: '{captions_filepath}' not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{captions_filepath}': {e}")
        return

    print(f"Processing '{results_filepath}'...")
    try:
        with open(results_filepath, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline()
            if not header.strip().lower().startswith("image_name| comment_number| comment"):
                print(f"Warning: '{results_filepath}' does not appear to have the expected header 'image_name| comment_number| comment'. Proceeding anyway.")

            for line_num, line in enumerate(f, 2): # Start line_num from 2 for data lines
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 3: # Expect at least 3 parts: image_name, comment_number, comment
                    image_name = parts[0].strip()
                    # We ignore parts[1] which is the comment_number
                    caption = parts[2].strip()
                    image_captions[image_name].add(caption)
                else:
                    print(f"Warning: Skipping malformed line {line_num} in '{results_filepath}': '{line}'")
    except FileNotFoundError:
        print(f"Error: '{results_filepath}' not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{results_filepath}': {e}")
        return

    print(f"Writing combined and formatted captions to '{output_filepath}'...")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # Sort image names for consistent output order
            for image_name in sorted(image_captions.keys()):
                # Convert set to list and sort captions for consistent numbering
                sorted_captions = sorted(list(image_captions[image_name]))
                for i, caption in enumerate(sorted_captions):
                    # Format: image_id#caption_number\tcaption_text
                    formatted_line = f"{image_name}#{i}\t{caption}\n"
                    outfile.write(formatted_line)
        print(f"Successfully combined and formatted captions to '{output_filepath}'.")
        print(f"Total unique images with captions: {len(image_captions)}")
    except Exception as e:
        print(f"An error occurred while writing to '{output_filepath}': {e}")

# --- Example Usage (assuming files are in the same directory as the script) ---
if __name__ == "__main__":
    # These file paths will refer to the files you uploaded
    captions_file = "captions.txt"
    results_file = "flickr30k_images/results.csv"
    output_file = "combined_flickr8k_style_captions.txt"

    combine_and_format_captions(captions_file, results_file, output_file)

    # Optional: Print the content of the generated output file
    print("\n--- Content of generated combined_flickr8k_style_captions.txt (first 20 lines) ---")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 20: # Print only first 20 lines for brevity
                    break
                print(line.strip())
    except FileNotFoundError:
        print("Output file not found (this indicates an error in the script).")

