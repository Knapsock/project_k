import os

# Folder to test
UPLOAD_DIR = "uploads"
FILENAME = "test_check.txt"
FILEPATH = os.path.join(UPLOAD_DIR, FILENAME)

# Step 1: Ensure uploads/ exists
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"[INFO] Created directory: {os.path.abspath(UPLOAD_DIR)}")
except Exception as e:
    print(f"[ERROR] Could not create upload folder: {e}")
    exit()

# Step 2: Attempt to write a file
try:
    with open(FILEPATH, "w") as f:
        f.write("This is a test file written by test_file_save.py.")

    # Step 3: Confirm the file was created
    if os.path.exists(FILEPATH):
        print(f"✅ File successfully written: {os.path.abspath(FILEPATH)}")
    else:
        print(f"❌ File missing after write: {FILEPATH}")

except Exception as e:
    print(f"❌ Exception occurred while writing file: {e}")
