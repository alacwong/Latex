import src.MachineLearning.Cluster as cl
import src.image.vision as vision
import src.image.image_process as im

path = "Enter path here"
image = im.read_image(path)
response = vision.read_image(im.image_to_byte(image))

math_set = {"+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
blocks = []
math_symbols = []
threshold = 0.5
for page in response.pages:
    for block in page:
        blocks.append(block)
        for paragraph in block.paragraph:
            for word in paragraph.words:
                for symbol in symbol.words:
                    if symbol.confidence < threshold or symbol.text in math_set:
                        math_symbols.append(symbol)

eps = cl.compute_epsilon(blocks)
[data, point_map] = cl.process_data(symbol)
cluster_map = cl.get_cluster(eps, data, point_map)
clusters = im.merge(cl.get_bounds_obj(cluster_map, point_map), cl.get_min_symbol(cluster_map, point_map), eps, image)

# Reannotate cluster

# recluster the annotation and then sort