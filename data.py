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

# Split by author/section instead
# Split on common patterns
sections = []
current_section = []
current_title = ""

lines = full_text.split('\n')

for line in lines:
    # Look for section headers (usually all caps, followed by content)
    if line.strip() and len(line.strip()) > 5 and line.strip().isupper() and len(line.strip().split()) < 10:
        # Save previous section
        if current_section and len(current_section) > 50:  # Only save substantial sections
            filename = os.path.join(output_dir, f"{current_title.replace(' ', '_').replace('/', '_')[:80]}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_section))
            print(f"Saved: {current_title[:60]} ({len(current_section)} lines)")
        
        current_title = line.strip()
        current_section = [line]
    else:
        if current_section or line.strip():
            current_section.append(line)

# Save final section
if current_section and len(current_section) > 50:
    filename = os.path.join(output_dir, f"{current_title.replace(' ', '_').replace('/', '_')[:80]}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(current_section))
    print(f"Saved: {current_title[:60]} ({len(current_section)} lines)")

print(f"\n✓ Finished!")
print(f"All files saved to {output_dir}")

# List what we created
print("\nFiles created:")
files = sorted(os.listdir(output_dir))
for file in files:
    filepath = os.path.join(output_dir, file)
    size = os.path.getsize(filepath) / 1e6
    print(f"  - {file} ({size:.1f} MB)")
