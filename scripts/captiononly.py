# File: extract_captions.py

input_file = "cleaned_captions.txt"
output_file = "captions_only.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            caption = parts[1]
            outfile.write(caption + "\n")

print("Captions extracted and saved to:", output_file)
