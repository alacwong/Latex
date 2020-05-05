# image processing library
# prepossess image before deploying to ML
from PIL import Image
import io
from PIL import ImageDraw
from _collections import deque
import math

input_length = 224
input_width = 224

colors = [
    "#ff0000",
    "#ffa500",
    "#ffff00",
    "#008000",
    "#0000ff",
    "#4b0082",
    "#ee82ee"
]
RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, PINK = 0, 1, 2, 3, 4, 5, 6
grid = {}


def read_image(path: str):
    """S
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


def img_resize(img, bound, min_symbol):
    """
    Resize image
    :param min_symbol: symbol to scale
    :param bound: bounding polygon
    :param img: image formatted in PIL
    :return: image resized to input size for ml prediction
    """
    width = min_symbol.boundingPoly.vertices[2].x - min_symbol.boundingPoly.vertices[0].x
    length = min_symbol.boundingPoly.vertices[2].y - min_symbol.boundingPoly.vertices[0].y

    ratio = min(input_width / width, input_length / length)

    return img.resize((int(ratio*(bound[2] - bound[0])), int(ratio*(bound[3] - bound[1]))), Image.ANTIALIAS)


def merge(cluster, image):
    """
    Merge list of clusters represented by 4 tuple into one image
    e[i] = (x1, y1, x2, y2)
    :return:white image with expressions pasted on
    """
    x = 0
    y = 0
    max_y = 0
    canvas = Image.new('RGB', (800, 1200), (255, 255, 255))
    q = deque()

    for e in cluster:
        if x + e[2] - e[1] > input_width:   # paste row
            canvas.paste(image, (x, y, image.width, image.height))
            if max_y < image.height:
                max_y = image.height
            q.append()


def draw_boxes(image, bounds):
    """
    Draw a border around the image using the hints in the vector list.
    Order of images found is by color
    """
    draw = ImageDraw.Draw(image)
    if bounds[0].normalized_vertices:
        width = image.width
        height = image.height
        for i in range(len(bounds)):
            draw.polygon([
                bounds[i].normalized_vertices[0].x * width, bounds[i].normalized_vertices[0].y * height,
                bounds[i].normalized_vertices[1].x * width, bounds[i].normalized_vertices[1].y * height,
                bounds[i].normalized_vertices[2].x * width, bounds[i].normalized_vertices[2].y * height,
                bounds[i].normalized_vertices[3].x * width, bounds[i].normalized_vertices[3].y * height],
                None, colors[i % len(colors)])
        return image
    else:
        for i in range(len(bounds)):
            draw.polygon([
                bounds[i].vertices[0].x, bounds[i].vertices[0].y,
                bounds[i].vertices[1].x, bounds[i].vertices[1].y,
                bounds[i].vertices[2].x, bounds[i].vertices[2].y,
                bounds[i].vertices[3].x, bounds[i].vertices[3].y],
                None, colors[i % len(colors)])
        return image

