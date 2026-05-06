import requests
import os

# Download the text file
url = "https://archive.org/stream/AnteNiceneFathersCompleteVolumesIToIX_201407/Ante-nicene%20fathers%20-%20complete%20volumes%20I%20to%20IX_djvu.txt"
output_dir = "/workspace/data"
os.makedirs(output_dir, exist_ok=True)

print("Downloading complete Ante-Nicene Fathers text...")
response = requests.get(url, timeout=30)
response.raise_for_status()

full_text = response.text
print(f"Downloaded {len(full_text):,} characters")

# Save as single file first
single_file_path = os.path.join(output_dir, "ante_nicene_complete.txt")
with open(single_file_path, 'w', encoding='utf-8') as f:
    f.write(full_text)
print(f"Saved complete text to {single_file_path}")

# Now split by volumes (look for "VOLUME" markers)
lines = full_text.split('\n')
current_volume = None
current_text = []
volume_count = 0

for line in lines:
    # Check if this is a volume header
    if 'VOLUME' in line.upper() and any(str(i) in line for i in range(1, 11)):
        # Save previous volume if it exists
        if current_volume and current_text:
            filename = os.path.join(output_dir, f"volume_{current_volume:02d}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_text))
            print(f"Saved volume {current_volume} ({len(current_text)} lines)")
            volume_count += 1
        
        # Start new volume
        current_volume = int(''.join(filter(str.isdigit, line.split()[0:3])))
        current_text = [line]
    else:
        if current_volume is not None:
            current_text.append(line)

# Save final volume
if current_volume and current_text:
    filename = os.path.join(output_dir, f"volume_{current_volume:02d}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(current_text))
    print(f"Saved volume {current_volume} ({len(current_text)} lines)")
    volume_count += 1

print(f"\n✓ Finished!")
print(f"Total volumes extracted: {volume_count}")
print(f"All files saved to {output_dir}")

# List what we created
print("\nFiles created:")
for file in sorted(os.listdir(output_dir)):
    filepath = os.path.join(output_dir, file)
    size = os.path.getsize(filepath) / 1e6
    print(f"  - {file} ({size:.1f} MB)")
