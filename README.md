# Реализация проекта для онлайн трансляции трекинга людей
Cервис поддерживает две модели - Bytetrack и IIM.

### Предварительные загрузки
Перед загрузкой необходимо заранее загрузить модели. 

# Загрузка из хаба
Чтоб стянуть докер-образ из докерхаба, выполните:
```
docker pull salos/tracking:v1.0
```

# Локально:

Сначала нужно загрузить модели. Для bytetrack они доступны по [ссылке](https://github.com/ifzhang/ByteTrack). Раздел Model Zoo. Загруженные модели необходимо положить в папку `detection_models/bytetrack/pretrained`.

Для IIM нужно переложить [файл](https://cloud.mail.ru/public/tDFV/nTQk76xrY/FDST-HR-ep_177_F1_0.969_Pre_0.984_Rec_0.955_mae_1.0_mse_1.5.pth) в `detection_models/iim/weights`,
предварительно переименовав его в `FDST-HR.pth` (это название дефолтное в настройках, но путь можно менять).


# Сборка
Чтобы собрать модель в docker необходимо запустить следующую команду:
```
docker build -t tracking .
```
Чтобы запустить образ для с необходимыми моделями нужно либо прописать из во флаге use_models, либо в конфиге (`config.yaml`):
```    
docker run --rm --gpus all -p 8132:8132 -e use_models=iim,bytetrack tracking python3 flask_app.py
```

Если недоступны GPU, то убрать флаг `--gpus all`.

Сервис должен стать доступен на http://localhost:8132/


