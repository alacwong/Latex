import os, io
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image

from pprint import pprint
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Alac Wang\Desktop\LatexTranslator\venv\credentials.json'


def read_image_test(client: types.image_annotator_pb2, content: bytes):
    """
    Reads the image from test directoryt and prints sentences
    :return None
    """
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)
    pages = response.full_text_annotation.pages
    text = ""
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                sentence = ""
                for word in paragraph.words:
                    s = ''.join([symbol.text for symbol in word.symbols])
                    sentence = sentence + " " + s
                text += sentence + "\n"


def get_image(test_dir_num: int, file_num: int):
    PATH = '../Test'
    path, dirs, files = next(os.walk(PATH))
    print(dirs)
    FILE = dirs[test_dir_num]
    test_images = os.listdir(PATH + "\\" + FILE)
    with io.open(os.path.join(PATH + "\\" + FILE, test_images[file_num]), 'rb') as image_file:
        content = image_file.read()
    return content


def debug_image(client: types.image_annotator_pb2, content):
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)
    pages = response.full_text_annotation.pages
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    w = ""
                    for symbol in word.symbols:
                        w += symbol.text
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))
                    print(w)
                    print("******")


def find_unknown(client: types.image_annotator_pb2, content, im):
    l = []
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)
    pages = response.full_text_annotation.pages
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                sentence = ""
                p_unknown = False
                for word in paragraph.words:
                    text = ""
                    for symbol in word.symbols:
                        if symbol.confidence < 0.5:
                            p_unknown = True
                            l.append(crop_bound(im, symbol.bounding_box))
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))
                        text += symbol.text
                    sentence += text + " "
                if p_unknown:
                    print('\t Sentence: {}'.format(sentence))
    return l


def get_image_pil(test_dir_num: int, file_num: int):
    PATH = '../Test'
    path, dirs, files = next(os.walk(PATH))
    print(dirs)
    FILE = dirs[test_dir_num]
    test_images = os.listdir(PATH + "\\" + FILE)
    print(PATH + "\\" + FILE, test_images[file_num])
    im = Image.open(fp=PATH + "\\" + FILE + "\\" + test_images[file_num], mode='r')
    # im.show()
    return im


def reannotate(image, symbol):
    client = vision.ImageAnnotatorClient()
    symbol1 = get_symbol(client, image, "el")
    symbol2 = get_symbol(client, image, "la")
    dict = {}
    dict[symbol.confidence] = symbol
    dict[symbol1.confidence] = symbol1
    dict[symbol2.confidence] = symbol2
    key = max(dict)
    return dict[key]

def read_image2(client, img):

    content = convert_byte(img)
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)
    pages = response.full_text_annotation.pages
    THRESHOLD = 0.5
    for page in pages:
        output = ""
        for block in page.blocks:
            phrase = ""
            for paragraph in block.paragraphs:
                sentence = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        word_text = ""
                        if symbol.confidence < THRESHOLD:
                            symbol = reannotate(convert_byte(crop_bound(img, symbol.bounding_box)),symbol)
                        word_text += symbol.text
                sentence += word_text + " "
            phrase += sentence + "\n"
        output += phrase + "***********************\n"
    return output

def convert_byte(img):
    roiImg = img.crop(box=None)
    content = io.BytesIO()
    roiImg.save(content, format='PNG')
    content = content.getvalue()
    return content

def get_symbol(client, content, language):
    image = vision.types.Image(content=content)
    response = client.text_detection(
        image=image,
        image_context={"language_hints": [language]},
    )
    pages = response.full_text_annotation.pages
    print(pages)

    max_symbol = None
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    max_confidence = 0
                    for symbol in word.symbols:
                        if symbol.confidence > max_confidence:
                            max_confidence = symbol.confidence
                            max_symbol = symbol

    return max_symbol

def crop_bound(image, boundingPoly):
    scale = 1.05  # 1%
    im = image.crop((boundingPoly.vertices[0].x // scale, boundingPoly.vertices[0].y // scale,
                     int(boundingPoly.vertices[2].x * scale), int(boundingPoly.vertices[2].y) * scale))
    return im



client = vision.ImageAnnotatorClient()
# image = get_image(0, 0)
im = get_image_pil(0, 0)
read_image2(client, im)
# x = find_unknown(client, image, im)
# img = get_image_pil(0, 0)
# roiImg = img.crop(box=None)
# imgByteArr = io.BytesIO()
# roiImg.save(imgByteArr, format='PNG')
# imgByteArr = imgByteArr.getvalue()




# debug_image(client, imgByteArr)
# read_image_test(client, 0, 1)
# print("*****************")
# read_image_test(client, 1, 2)
# print("*****************")
# read_image_test(client, 1, 3)

# yoinked stackoverflow code
# response = client.text_detection(
#     image=image,
#     image_context={"language_hints": ["bn"]},  # Bengali
# )
