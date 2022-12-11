# Реализация проекта для онлайн трансляции трекинга людей
Cервис поддерживает две модели - Bytetrack и IIM.



# Загрузка из хаба
Чтоб стянуть докер-образ из докерхаба, выполните:
```
docker pull salos/tracking:v1.0
```

# Локально:

### Предварительные загрузки
Перед загрузкой необходимо заранее загрузить модели. 

Для bytetrack они доступны по [ссылке](https://github.com/ifzhang/ByteTrack). Раздел Model Zoo. Загруженные модели необходимо положить в папку `detection_models/bytetrack/pretrained`.

Для IIM нужно переложить [файл](https://cloud.mail.ru/public/tDFV/nTQk76xrY/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth) в `detection_models/iim/weights`.


# Сборка
Чтобы собрать модель в docker необходимо запустить следующую команду:
```
docker build -t tracking .
```
Чтобы запустить образ для с необходимыми моделями нужно либо прописать их во флаге use_models, либо в конфиге `config.yaml` (чтение из конфига будет производиться, если переменная среды `use_models` не указана):
```    
docker run --rm --gpus all -p 8132:8132 -e use_models=iim,bytetrack tracking python3 flask_app.py
```

Если недоступны GPU, то убрать флаг `--gpus all`.

Сервис должен стать доступен на http://localhost:8132/


