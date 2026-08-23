import glob

from PIL import Image, ImageOps

for path in glob.glob("data/spot/*/*.jpg"):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # Rotates the image according to the Exif tag
    img.save(path)  # Overwrites the original
