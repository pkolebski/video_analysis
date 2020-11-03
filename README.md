# MIO-TCD dataset
yolov4

1. download and unpack [MIO-TCD Localization](http://podoce.dinf.usherbrooke.ca/static/dataset/MIO-TCD-Localization.tar) in /data/
2. clone [Yolov4](https://github.com/AlexeyAB/darknet)
3. build Yolov4
4. run ./preparation_train_yolov4-tiny.py
5. run ./darknet/darknet detector train yolo_v4_train_params/obj.data yolo_v4_train_params/yolov4-tiny-custom.cfg yolo_v4_train_params/yolov4-tiny.conv.29 

 