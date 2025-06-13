## Инструкция для МИФИ
## # Создание кластера

Тут полноценная инструкция по созданию кластера: https://yandex.cloud/ru/docs/data-proc/operations/cluster-create

Там ограничения: 200 Гб  физ памяти, 32 ядра

Настройка окружения:
```
sudo apt-get update
sudo pip install s3cmd
s3cmd --configure
```
Дальше вступает в силу инструкция: https://yandex.cloud/ru/docs/storage/tools/s3cmd

Она позволяет создать s3-хранилище, где нужно будет дополнительно создать бакет для хранения всего что хочется

Дальше закинул в хранилище geesefs-linux-amd64 - эта штука нужная для монтирование хранилища как локальной файловой системы
```
s3cmd get s3://image-storage-bucket/geesefs-linux-amd64
sudo apt-get update
```
Дальше нужно экспортировать ключи (это тоже есть в инструкции). Пример:
```
export КЛЮЧ_ДОСТУПА=111111111111111111
export КЛЮЧ_СЕКРЕТНЫЙ=2222222222222222222222
```
Дальше создаём директорию и хотим монтировать туда хранилище:
```
mkdir mounter
chmod a=rwx geesefs-linux-amd64
./geesefs-linux-amd64 image-storage-bucket /home/ubuntu/mounter
```

Должны получить уведомление что всё хорошо сработало :-)

Дальше установка всякого питоновского окружения
```
python3 -m venv my_env
source /my_env/bin/activate
source my_env/bin/activate
sudo apt-get install openslide-tools
sudo apt-get update
pip install openslide-python
sudo apt-get update
pip install numpy==1.24.4
pip install torch==2.2.1
pip install albumentations==1.4.18
pip install segmentation-models-pytorch==0.3.3
pip install pandas==2.0.3
pip install cv2geojson==0.0.9
pip install matplotlib 2.7.2
pip install matplotlib 3.7.2
pip install matplotlib
```
Такое нужно сделать на каждом узле.

Прикрепленные файлы:
start.py - Код для запуска распределенного обучения, запускается просто со своего локального компьютера. Аналогичный можно сделать для запуска применения.

htohto.py - код для распределенного обучения (тут явно стоит проверить, явно надо что-то не так или неправильно, не успел доделать)

import.py - код для распределенного применения модели к изображению