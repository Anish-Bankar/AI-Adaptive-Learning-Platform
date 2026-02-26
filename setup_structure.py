import os
import shutil

# Root project folder
BASE_DIR = os.getcwd()

# Folder structure
folders = [
    "docs",
    "docs/demo_screenshots",
    "sample_data",
    "assets",
    "config",
]

# Create folders
for folder in folders:
    path = os.path.join(BASE_DIR, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")

# Move JSON profile files to config folder
json_files = [
    "advanced_student_profile.json",
    "sppu_student_profile.json",
    "universal_student_profile.json",
]

for file in json_files:
    src = os.path.join(BASE_DIR, file)
    dst = os.path.join(BASE_DIR, "config", file)

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {file} → config/")
    else:
        print(f"{file} not found (skipped)")

# Create README if not exists
readme_path = os.path.join(BASE_DIR, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write("# AI Driven Adaptive Platform\n")
    print("Created README.md")

print("\n✅ Project structure setup complete.")