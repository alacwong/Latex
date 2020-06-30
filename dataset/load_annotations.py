import csv
annotations = {
    'TRAIN': {},
    'TEST': {}
}

with open('export_data-latex4-2020-06-07T16_00_23.659Z_image_object_detection_1.csv', newline='', encoding='utf-8-sig') as f:
    data = list(csv.reader(f))


for elem in data:
    if elem[0] == 'VALIDATION':
        data_type = 'TEST'
    else:
        data_type = elem[0]
    img = elem[1].replace('-2020-06-05T02:28:12.084Z', '')
    if elem[1] != img:
        annotations[data_type][img.replace('gs://latex5/', '')] = {
            'class': elem[2],
            'top-left': (elem[3], elem[4]),
            'top-right': (elem[5], elem[6]),
            'bottom-right': (elem[7], elem[8]),
            'bottom-left': (elem[9], elem[10])
        }
print(annotations)



