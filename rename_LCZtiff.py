from dotenv import load_dotenv
import os

load_dotenv()

lcz_dir = os.getenv("LCZ_DIR")

for location in os.listdir(lcz_dir): # location is the folder name
    location_dir = os.path.join(lcz_dir, location)
    for file in os.listdir(location_dir): # file is the tiff file
        if file.endswith(".tif"):
            raw_name = file.split(".")[0]
            break
    if (raw_name == location):
        continue
    #iterate over all files in the location_dir and replace raw_name string with location
    for root, dirs, files in os.walk(location_dir):
        for file in files:
            new_name = file.replace(raw_name, location)
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
