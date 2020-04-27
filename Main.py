import os, io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Alac Wang\Desktop\LatexTranslator\venv\credentials.json'


def read_image_test(client , test_dir_num: int, file_num: int):
    """
    Reads the image from test directoryt and prints sentences
    :return None
    """

    PATH = 'Test'
    path, dirs, files = next(os.walk(PATH))
    FILE = dirs[test_dir_num]
    test_images = os.listdir(PATH + "\\" + FILE)
    with io.open(os.path.join(PATH + "\\" + FILE, test_images[file_num]), 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)

    pages = response.full_text_annotation.pages
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                y = ""
                for word in paragraph.words:
                    s = ''.join([symbol.text for symbol in word.symbols])
                    y = y + " " + s
                print(y)



client = vision.ImageAnnotatorClient()
read_image_test(client, 1, 1)
print("*****************")
read_image_test(client, 1, 2)
print("*****************")
read_image_test(client, 1, 3)