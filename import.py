import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from cv2geojson import find_geocontours, GeoContour
from tqdm import tqdm
import copy
from pathlib import Path
from typing import List, Dict
import numpy as np
import os
import json
import logging
import pandas as pd
from openslide import open_slide
import numpy as np
import time
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from shapely.ops import unary_union
import sys

Image.MAX_IMAGE_PIXELS = None
DIRECTORY_TO_WATCH = "/home/ubuntu/mounter/images/"
DIRECTORY_FOR_RESULT = "/home/ubuntu/mounter/result/"
IGNORE_PATTERN = "__COPYING__"
HOME_DIRECTORY = "/home/ubuntu/mounter/images/"
MOUNTING_DIRECTORY = "/home/ubuntu/mounter/"

SCAN_INTERVAL = 20
SIZE = 1024
TILE_SIZE = 256
LEVEL = 1
LEVEL_SAVING = 1
M1 = 2
N1 = 3  # У каждого узла свой N1 (1, 2 или 3)
AMOUNT = 3

settings = {
    'cell_type': {'pseudo': 'Pseudo_Unet_mobilenet_v2.pt', 'hurtle': 'hurtle_Unet-mobilenet_v2_dice-0_895.pt', 'binuclears': 'binuclears_path.pth'},
    'tile_size': TILE_SIZE,
    'image_size': SIZE,
    'level': LEVEL,
}

class SegmentationModel():
    def __init__(self, cell_type: str='hurtle'):
        self._model = None
        self.cell_type = cell_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self, path: str) -> None:
        path2 = Path(path)
        if self.cell_type not in settings['cell_type']:
            raise ValueError('No such cell type')
        model_path = path2 / settings['cell_type'][self.cell_type]
        self._model = torch.load(model_path, map_location=self.device)

    def preprocessing(self, image: np.array) -> Dict[str, torch.Tensor]:
        val_transform = A.Compose([
            A.Resize(settings['tile_size'], settings['tile_size']),
            A.Normalize(),
            ToTensorV2(),
        ])
        return val_transform(image=image)

    def process_batch(self, batch_tiles: List[torch.Tensor], batch_names: List[tuple]) -> List[Dict[str, Dict[str, List[List[int]]]]]:
        batch = torch.stack(batch_tiles)
        with torch.no_grad():
            outputs = self._model(batch).argmax(axis=1).cpu().detach().numpy()
        features = []

        for i in range(len(batch_names)):
            col, row = batch_names[i]
            output = outputs[i]
            result = (output * 255).astype(np.uint8)

            geocontours = find_geocontours(result, mode='imagej')[0]
            geocontours.scale_up(ratio=settings['image_size']/settings['tile_size'], 
                                 offset=(col*settings['image_size'], row*settings['image_size']))

            feature = geocontours.export_feature(color=(255, 0, 0), label='rectangle')
            if len(feature['geometry']['coordinates']) > 1:
                feature['geometry']['coordinates'] = feature['geometry']['coordinates'][1:]

                for i in range(len(feature['geometry']['coordinates'])):
                    one_feature = copy.deepcopy(feature)
                    one_feature['geometry']['coordinates'] = [one_feature['geometry']['coordinates'][i]]
                    area = GeoContour(geometry=one_feature['geometry']).area()
                    if area > 200:
                        features.append(one_feature)
                        self.count += 1
        return features

    def cut_segment_batch(self, slide: open_slide, level: int, image_size: int, batch_size: int=4) -> List[Dict[str, Dict[str, List[List[int]]]]]:
        global M1, N1
        cols, rows = np.array(slide.level_dimensions[level]) // image_size
        batch_tiles = []
        batch_names = []
        features = []

        # Определяем границы обрабатываемой части (1/3 изображения)
        total_cols = cols
        part_cols = total_cols // AMOUNT
        start_col = M1 * part_cols
        end_col = N1 * part_cols if N1 != AMOUNT else total_cols  # Последний узел берет остаток

        for col in tqdm(range(start_col,end_col)):

            for row in range(rows):
                orig_img = slide.read_region((col * image_size * 4**level, row * image_size * 4**level), level,
                                            size=(image_size, image_size))
                img = np.array(orig_img.convert('RGB'))
                M = img.shape[0]//4
                N = img.shape[1]//4
                batch_tiles = [self.preprocessing(img[x:x+M,y:y+N])['image'].to(self.device) for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
                batch_names = [(col*4+c, row*4+r) for r in range(4) for c in range(4)]
                features += self.process_batch(batch_tiles, batch_names)

        if len(batch_tiles) > 0:
            features += self.process_batch(batch_tiles, batch_names)

        return features

    def predict(self, path: str) -> List[Dict[str, Dict[str, List[List[int]]]]]:
        slide = open_slide(path)
        self.count = 0
        features = self.cut_segment_batch(
            slide,
            level=settings['level'],
            image_size=settings['image_size']
        )
        return features

def scaling_factor_counting(image_path, level_of_saving):
    slide_for_scale = open_slide(image_path)
    scale_factor = slide_for_scale.level_downsamples[level_of_saving]

    return scale_factor

def save_partial_output_image(image_path, save_path, level_of_saving):
    global N1,M1
    slide = open_slide(image_path)
    level_width, level_height = slide.level_dimensions[level_of_saving]
    print(level_width, level_height)

    scale_factor = scaling_factor_counting(image_path, level_of_saving)

    total_width_level0 = slide.dimensions[0]

    part_width_level0 = total_width_level0 // AMOUNT
    start_x_level0 = M1 * part_width_level0
    end_x_level0 = N1 * part_width_level0

    # Переводим в координаты выбранного уровня
    start_x_scaled = int(start_x_level0 / scale_factor)
    end_x_scaled = int(end_x_level0 / scale_factor)
    region_width_scaled = end_x_scaled - start_x_scaled

    alt = 0

    num_of_chunks = math.ceil((region_width_scaled)/65500)
    for i in range(num_of_chunks):
        width = min(65500, (end_x_scaled - start_x_scaled) - i*65500)
        # Вырезаем свою часть
        partial_image = slide.read_region(
            (start_x_level0, 0),
            level_of_saving,
            size=(region_width_scaled, level_height)
        ).convert('RGB')

        alt += width
        # Сохраняем свою часть
        partial_image.save(f'{save_path}output_image_{level_of_saving}_{N1}.jpg')
        print(f"Сохранена часть {N1} изображения: {save_path}output_image_{level_of_saving}_{N1}.jpg")

def count_chunks(image_path, level_of_saving):
    global AMOUNT
    slide = open_slide(image_path)
    level_width, level_height = slide.level_dimensions[level_of_saving-1]

    print(level_width)
    cnt = math.ceil(level_width / 65500 / AMOUNT)
    return cnt

def generate_list(image_path, level_of_saving):
    global AMOUNT

    slide = open_slide(image_path)
    level_width, level_height = slide.level_dimensions[level_of_saving-1]

    part_width = level_width / AMOUNT
    num_of_chunks_per_part = math.ceil(part_width/65500)

    list_of_width = []

    for _ in range(AMOUNT):
        remaining = part_width
        for _ in range(num_of_chunks_per_part - 1):
            list_of_width.append(65500)
            remaining -= 65500
        list_of_width.append(round(remaining,3))

    return list_of_width

def find_files_in_directory(directory):
    directory = Path(directory)
    geojson_files = list(directory.glob('*.geojson'))

    image_extensions = ['.jpg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f'*{ext}'))

    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON files found in {directory}")
    if not image_files:
        raise FileNotFoundError(f"No image files found in {directory}")
    if len(image_files) > 1:
        print(f"Warning: Multiple image files found, using {image_files[0]}")

    return geojson_files, image_files[0]

def merge_geojsons(geojson_files, output_file):
    merged_features = []

    for file in geojson_files:
        with open(file, 'r') as f:
            data = json.load(f)
            merged_features.extend(data['features'])

    merged_geojson = {
        "type": "FeatureCollection",
        "features": merged_features
    }

    with open(output_file, 'w') as f:
        json.dump(merged_geojson, f)

    return merged_geojson

def process_files(directory, scale_factor):
    """Основная функция обработки"""

    geojson_files, image_path = find_files_in_directory(directory)
    print(f"Found {len(geojson_files)} GeoJSON files: {[f.name for f in geojson_files]}")
    print(f"Using image: {image_path.name}")

    output_geojson_path = directory + 'merged.geojson'
    image_path_save = directory + 'result.jpg'

    merged_geojson = merge_geojsons(geojson_files, output_geojson_path)

    img = mpimg.imread(image_path)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")  # Width x Height

    gdf = gpd.read_file(output_geojson_path)
    print(f"GeoJSON bounds before scaling: {gdf.total_bounds}")

    gdf['geometry'] = gdf['geometry'].scale(
        xfact=1.0/float(scale_factor),
        yfact=1.0/float(scale_factor),
        origin=(0, 0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])

    gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    ax.set_aspect('equal')  # Сохраняем пропорции
    ax.set_title("Наложение разметки на изображение")

    plt.savefig(image_path_save, dpi=300, bbox_inches='tight')
    print(f"Result saved to: {image_path_save}")

try:
    previous_files = []
    logging.info(f"Directory monitoring: {DIRECTORY_TO_WATCH}")

    while True:
        try:
            start_time = time.time()
            files = os.listdir(DIRECTORY_TO_WATCH)
            current_files = [f for f in files if not re.search(IGNORE_PATTERN, f)]
            new_files = [item for item in current_files if item not in previous_files]

            for file_path in new_files:
                new_name = file_path.split('/')[-1]
                logging.info(f'Working with file: {new_name}')

                # Сохраняем свою часть изображения
                save_partial_output_image(
                    f'{DIRECTORY_TO_WATCH}/{new_name}',
                    DIRECTORY_FOR_RESULT,
                    LEVEL_SAVING
                )

                # Обработка модели и сохранение GeoJSON
                model = SegmentationModel(cell_type='hurtle')
                model.load(MOUNTING_DIRECTORY)
                features = model.predict(f'{DIRECTORY_TO_WATCH}/{new_name}')

                with open(f'{DIRECTORY_FOR_RESULT}/{new_name}_{N1}.geojson', 'w') as f:
                    json.dump(
                        {"type": "FeatureCollection", "features": features}, 
                        f,
                        indent=4
                    )

                # Только первый узел (N1 == 1) собирает финальное изображение и geojson
                if N1 == 1:
                    while True:
                        files_geojson = [f for f in os.listdir(DIRECTORY_FOR_RESULT) 
                                      if f.endswith('.geojson') and f.startswith(new_name)]
                        if len(files_geojson) == AMOUNT:  # Ждем, пока все узлы сохранят свои части
                            break
                        time.sleep(10)

                    # Собираем все части изображения
                    img_parts = []
                    for i in range(1, AMOUNT + 1):
                        img_part = Image.open(f'{DIRECTORY_FOR_RESULT}output_image_{LEVEL_SAVING}_{i}.jpg')
                        img_parts.append(img_part)

                    # Объединяем части в одно изображение
                    total_width = sum(img.width for img in img_parts)
                    max_height = max(img.height for img in img_parts)
                    final_img = Image.new('RGB', (total_width, max_height))

                    x_offset = 0
                    for img in img_parts:
                        final_img.paste(img, (x_offset, 0))
                        x_offset += img.width

                    # Сохраняем финальное изображение
                    final_img.save(f'{DIRECTORY_FOR_RESULT}/{new_name}_final.jpg')
                    print(f"Финальное изображение сохранено: {DIRECTORY_FOR_RESULT}/{new_name}_final.jpg")

                    # Объединяем GeoJSON файлы
                    merged_features = []
                    for i in range(1, AMOUNT + 1):
                        geojson_path = f'{DIRECTORY_FOR_RESULT}/{new_name}_{i}.geojson'
                        with open(geojson_path, 'r') as f:
                            data = json.load(f)
                            merged_features.extend(data['features'])

                    merged_geojson = {
                        "type": "FeatureCollection",
                        "features": merged_features
                    }

                    merged_geojson_path = f'{DIRECTORY_FOR_RESULT}/{new_name}_merged.geojson'
                    with open(merged_geojson_path, 'w') as f:
                        json.dump(merged_geojson, f, indent=4)
                    print(f"Объединенный GeoJSON сохранен: {merged_geojson_path}")

                    #scale_factor = scaling_factor_counting(f'{DIRECTORY_TO_WATCH}/{new_name}', LEVEL_SAVING)
                    #process_files(DIRECTORY_FOR_RESULT, scale_factor=scale_factor)

                sys.exit(0)
                previous_files = current_files

            end_time = time.time()
            print(f'Time of processing: {end_time - start_time}')
            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            logging.error(f'Error log: {e}')
except KeyboardInterrupt:
    logging.info('A completion signal has been received')
logging.info('Programm is completed')
