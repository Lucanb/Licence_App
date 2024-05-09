import os
from PIL import Image

def resize_and_save_images(source_dir, target_dir, new_size=(250, 250)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(source_dir, filename)
            with Image.open(full_path) as img:
                resized_image = img.resize(new_size, Image.Resampling.LANCZOS)
                save_path = os.path.join(target_dir, filename)
                
                resized_image.save(save_path)

# resize_and_save_images('dataset/images', 'dataset/img1')
# resize_and_save_images('dataset/masks/img', 'dataset/mask1')
resize_and_save_images('images', 'img1')