# Реализация проекта для онлайн трансляции трекинга людей
Cервис поддерживает две модели - Bytetrack и IIM.

### Для bytetrack
Перед загрузкой необходимо заранее загрузить модели. Они доступны по [ссылке](https://github.com/ifzhang/ByteTrack). Раздел Model Zoo. Загруженные модели необходимо положить в папку `detection_models/bytetrack/pretrained`.
Чтобы собрать модель в docker необходимо запустить следующую команду:
```
docker build -t tracking .
```
Чтобы запустить образ для bytetrack:
```    
docker run --rm -p 8132:8132 --gpus all -e use_model=bytetrack tracking
```


### Для IIM

Для использования IIM нужно переложить [файл](https://cloud.mail.ru/public/tDFV/nTQk76xrY/FDST-HR-ep_177_F1_0.969_Pre_0.984_Rec_0.955_mae_1.0_mse_1.5.pth) в `detection_models/iim/weights`,
предварительно переименовав его в `FDST-HR.pth` (это название дефолтное в настройках, но путь можно менять).

Чтобы собрать модель в docker необходимо запустить следующую команду:
```
docker build -t tracking .
```

Чтобы запустить образ для IIM:
```    
docker run --rm -p 8132:8132 --gpus all -e use_model=iim tracking
```


Сервис должен стать доступен на http://localhost:8132/


