import shutil
import os

# Path to the original folder
original_folder = 'environment_0'

# Loop to copy the folder 100 times
for i in range(1, 101):
    # Create new folder name by incrementing the number
    new_folder = 'environment_' + str(i)

    # Copy the original folder to the new folder
    shutil.copytree(original_folder, new_folder)
