# MIO-TCD dataset

1. download and unpack [MIO-TCD Localization](http://podoce.dinf.usherbrooke.ca/static/dataset/MIO-TCD-Localization.tar) in /data/
    ```bash
    .
    ├── configs
    ├── data
    │   └── MIO-TCD-Localization
    │       ├── test
    │       └── train
    ├── ...
    ...
    ```

2. run `python ./preparation_train_yolov4-tiny.py`
3. clone [Yolov4](https://github.com/AlexeyAB/darknet)
    ```bash
    git clone https://github.com/AlexeyAB/darknet
    ```
    ```bash
    .
    ├── configs
    ├── data
    │   └── MIO-TCD-Localization
    │       ├── test
    │       └── train
    ├── 
   ..
    └── yolo_train_MIO_TCD
        ├── darknet
        │   ├── 3rdparty
        │   ...
        └── yolo_v4_train_params

   ```
4. build Yolov4
    ```bash
    cd darknet
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    sed -i 's/GPU=0/GPU=1/' Makefile
    sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    make
    chmod +x ./darknet
    ```
5. train

    `
    ./darknet detector train ../yolo_v4_train_params/obj.data ../yolo_v4_train_params/yolov4-tiny-custom.cfg ../yolo_v4_train_params/yolov4-tiny.conv.29 -map
    `
    
    continue training 
     `
     ./darknet detector train ../yolo_v4_train_params/obj.data ../yolo_v4_train_params/yolov4-tiny-custom.cfg backup/yolov4-tiny-custom_last.weights -map
     `
     