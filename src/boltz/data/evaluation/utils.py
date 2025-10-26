import os

def get_subdirectories(directory_path):
    # Get all sub dir.
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return subdirectories