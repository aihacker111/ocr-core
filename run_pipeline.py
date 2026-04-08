from PIL import Image
img    = Image.open("page.png")
result = pipeline.run_image(img)
print(result.formatted)