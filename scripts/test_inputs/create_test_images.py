from PIL import Image, ImageDraw
import os
os.makedirs('scripts/test_inputs', exist_ok=True)

# copy a dataset image if available
try:
    img = Image.open('Brain_Tumor_Datasets/Brain_Tumor_Datasets/test/YES/yes_1892.jpg')
    img.save('scripts/test_inputs/dataset_yes_1892.jpg')
except Exception as e:
    print('Could not copy dataset image', e)

# create a large non-square RGBA image (simulate external MRI scan)
W, H = 1400, 800
img2 = Image.new('RGBA', (W,H), (20,28,48,255))
d = ImageDraw.Draw(img2)
# draw some faint synthetic 'tumor' blob
d.ellipse((780,260,940,420), fill=(255,180,160,190))
d.ellipse((870,300,960,390), fill=(255,220,200,160))
img2.save('scripts/test_inputs/large_non_square_rgba.png')

# create a grayscale image (single channel)
img3 = Image.new('L', (512,512), 18)
d3 = ImageDraw.Draw(img3)
d3.ellipse((200,160,330,290), fill=220)
img3.save('scripts/test_inputs/grayscale_512.png')

print('Created test inputs: dataset_yes_1892.jpg (if present), large_non_square_rgba.png, grayscale_512.png')
