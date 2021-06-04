# Anomaly Crossing: New Horizons for Cross-domain Few-shot Video Anomaly Detection

This repository is the official implementation of Anomaly Crossing: New Horizons for Cross-domain Few-shot Video Anomaly Detection. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on DoTA, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
## Pre-trained Models

You can download pretrained models here:

- [IG65M+DAM](https://drive.google.com/mymodel.pth) trained on DoTA as the reference domain. 

## Results

Our model achieves the following performance on :

### [2-way 5 shot result on DoTA]

| Model name         | Accuracy for all types  | 
| ------------------ |-------------------------| 
| Anomaly Crossing   |            0.81         | 

