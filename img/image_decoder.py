import os
import io
import struct
import numpy as np
from PIL import Image

# Path to the .bin file
bin_file = 'C:\\Users\\initi\\OneDrive\\Documents\\GitHub\\facenet-pytorch-CN\\lfw\\lfw.bin'
output_dir = 'C:\\Users\\initi\\OneDrive\\Documents\\GitHub\\facenet-pytorch-CN\\lfw'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the .bin file
with open(bin_file, 'rb') as f:
    while True:
        # Read the image header (example: 4 bytes for length)
        img_len = f.read(4)
        if not img_len:
            break
        img_len = struct.unpack('i', img_len)[0]
        
        # Read image data
        img_data = f.read(img_len)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        
        # Convert to an image
        img = Image.open(io.BytesIO(img_array))
        
        # Save the image
        img_name = f'image_{f.tell()}.jpg'
        img.save(os.path.join(output_dir, img_name))

print(f"Images saved to {output_dir}")
