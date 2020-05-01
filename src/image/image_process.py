# image processing library
from PIL import Image
import io
input_length = 5
input_width = 5


def read_image(path: str):
    """
    Read file from path
    :param path: of file
    :return: image file (PIL)
    """
    return Image.open(path, mode="r")


def image_to_byte(img):
    """
    Convert img to bytes
    :param img: image file (PIL)
    :return byte representation of image file:
    """
    img2 = img.crop(box=None)
    byte_arr = io.BytesIO()
    img2.save(byte_arr, format='PNG')
    return byte_arr.getvalue()


def crop_img(image, bound):
    """
    Crop image on bounds given from vision api and PIL formatted image
    :param image: image file (PIL)
    :param bound: bounding polygon from vision api
    :return:
    """
    scale = 1.01  # 1%
    return image.crop((bound.vertices[0].x // scale, bound.vertices[0].y // scale,
                       int(bound.vertices[2].x * scale), int(bound.vertices[2].y) * scale))


def img_resize(img):
    """
    Resize image
    :param img:
    :return:
    """
    print("resizing image")
