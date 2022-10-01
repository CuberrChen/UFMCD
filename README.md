# UFMCD

- Baseline: Supervised only

|SSL|NetWork |iters|opt|image_size|batch_size|lr|dataset|label_ratio|memory|card|best mIou|weight|
|:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: |
|Base|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.5|16G|1*V100|77.00|[https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47](https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47)|
|Base|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.25|16G|1*V100|75.97|[https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47](https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47)|
|Base|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.125|16G|1*V100|74.85|[https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47](https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47)|
|Base|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.0625|16G|1*V100|73.23|[https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47](https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47)|
|Base|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.03125|16G|1*V100|70.80|[https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47](https://pan.baidu.com/s/1Go3Vc-7ZNc899VaSik77yw?pwd=im47)|

- UFMCD: Uncertainty-Guided Feature Mixing and Cross-Decoupled Pseudo Supervision for Semi-Supervised Semantic Segmentation(ICARCV, 2022)

|SSL|NetWork |iters|opt|image_size|batch_size|lr|dataset|label_ratio|memory|card|best mIou|weight|
|:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: |
|UFMCD|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.5|28G|1*V100|78.11|[https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr](https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr)|
|UFMCD|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.25|28G|1*V100|78.08|[https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr](https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr)|
|UFMCD|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.125|28G|1*V100|77.52|[https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr](https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr)|
|UFMCD|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.0625|28G|1*V100|76.95|[https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr](https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr)|
|UFMCD|Deeplabv2-Res101|20k|SGD|321x321|8|0.001|Pascal VOC2012+Aug|0.03125|28G|1*V100|76.92|[https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr](https://pan.baidu.com/s/1qgyR8TizU9-KRFTt1jA_Mw?pwd=dzfr)|


### 1. Requirements
Due to the computing resources factor, the project implementation is based on the paddle framework and if you are not familiar with it, you may need some time to realize it before you start.
```bash
pip install -r requirements.txt
```

### 2. Dataset
Please place pascal voc 2012 with aug dataset in `data/pascalvoc/VOCdevkit/`

The details can found at `paddleseg/datasets/voc.py`

### 3. Train
```bash
python train.py --config configs/deeplabv2/deeplabv2_resnet101_os8_voc_semi_321x321_20k.yml --device=0 --label_ratio 0.125 --ssl_method UFMCD --num_workers 0 --use_vdl --do_eval --save_interval 1000 --save_dir deeplabv2_res101_voc_0.125_20k
```

### 4. Val
```bash
python val.py --config configs/deeplabv2/deeplabv2_resnet101_os8_voc_semi_321x321_20k.yml --model_path deeplabv2_res101_voc_0.125_20k/best_model/model.pdparams
```

### Acknowledgement
The code is highly based on the [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg). 
Many thanks for their work.
Since I'm busing with job hunting, the repository will be further improved in the future.




