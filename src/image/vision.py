# library for reading images using vision api
import os
from google.cloud import vision
import src.image.image_process as im

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'credentials.json'
client = vision.ImageAnnotatorClient()


def read_image(path: str):
    """
    Return text annotated image
    :param path: path to image
    :return: api annotated image
    """
    content = im.read_image(path)
    content = im.image_to_byte(content)
    image = vision.types.Image(content=content)
    return client.document_text_detection(image=image)


def read_objects(path: str):
    """
    Find objects in image
    :param path: path to image
    :return list of objects in image:
    """
    content = im.read_image(path)
    content = im.image_to_byte(content)
    image = vision.types.Image(content=content)
    return client.object_localization(
        image=image).localized_object_annotations
