# Anomaly Crossing: New Horizons for Cross-domain Few-shot Video Anomaly Detection

This repository is the official implementation of Anomaly Crossing: New Horizons for Cross-domain Few-shot Video Anomaly Detection. 

## Requirements

Main requirements:

- Python3
- pytorch1.1+
- PIL

For the installation, you need to install conda. The environment may contain also unnecessary packages.

```setup
#Create the environment with the command
conda env create -f anomalyCrossing.yml

#Then you can activate the environment with the command
conda activate anomalyCrossing
```

## Datasets

- Detection of Traffic Anomaly (DoTA): Download the dataset from its official [GitHub Repo](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly).
- UCF-Crime: Download the dataset [here](https://webpages.uncc.edu/cchen62/dataset.html). Additional annotations is provided in MCPM/datasets/ucf_crime_more_annotations.

## Domain Adaptation Module (DAM)

Codes about DAM are in DAM folder. We use DoTA as the default dataset in this instruction.

Go to the DAM folder.

```
cd DAM
```

### Pre-trained Models

You can download pretrained model here to skip the DAM:

- [IG65M+DAM](https://drive.google.com/file/d/1MH_IZclrdHmfz0IMqLqmBfmrz-q6rP8P/view?usp=sharing) trained on DoTA as the reference domain. 

### Create Split Files

You can directly use our split files in **datasets/settings/dota_seq** and skip this step.



Create split files for DAM. Set the following args **before** running:

- `SETTING_PATH`: Path to dataset setting files.
- `BASE_PATH`: Path to dataset files.
- `FRAME_PATH`: Path to image files.
- `ANNOT_PATH`: Path to annotation files.
- `TRAIN_SP`: Train split file.
- `VAL_SP`: Validation split file.

```train
python create_data_split_dota_seq.py --settings SETTING_PATH --base-path BASE_PATH --frame-path FRAME_PATH --annot-path ANNOT_PATH --train-split-file TRAIN_SP --val-split-file VAL_SP --normal
```

Then, you will get 3 split files for train, val, and all in you `SETTING_PATH`. We will use  `all_rgb_split1.txt` in most cases.

### Apply DAM

Go to the source folder.

```cmd
cd src
```

The default running parameters can be changed in `option.py`.

Then, adapt the model. Set the following args **before** running:

- `FRAME_PATH`: Full path to image files.
- `OUTPUT_PATH`: Path to save the checkpoint of model output by DAM.
- `SPLIT_FILE`: Path with file name of your split file.
- `BASE_PATH`: Path to dataset files.
- `FRAME_PATH`: Path to image files.

```
python main_ddp.py --method dota --arch r2p1d_lateTemporal --ssl_arch rgb_r2plus1d_8f_34_encoder --pt_dataset dota --pt_root BASE_PATH --pt_train_list SPLIT_FILE --ssl_output OUTPUT_PATH --ssl_width 224 --ssl_height 224 --pt_data_length 8 --ssl_frame_path FRAME_PATH 
```

## Meta Context Perception Module (MCPM) and Evaluation

Codes about MCPM as well as the evaluation are in the MCPM folder. We use DoTA as the default dataset in this instruction.

Go to the MCPM folder.

```
cd MCPM
```

### Create Split Files

You can directly use our split files in **datasets/settings/dota** and skip this step.

Use **datasets/create_data_splits_dota.py** to create a new split. Arguments are same as the splitting codes for DAM except the `normal`.

### Save the video context features

Change the parameters at the beginning of the `main` function of `save_video_ctx_vectors.py` and run it.

```eval
python save_video_ctx_vectors.py
```
### Evaluate the performance with MCPM

For evaluation, run `test_meta_model_STGCN.py` with the following arguments:

-   `--dataset`: dataset for evaluation. 
-  `--settings` : path to dataset setting files.
-   `--ctx-path` path to dataset files.
-   `--phase`: phase of the split file, e.g. train, val or all.
-   `--name-pattern-ctx`: name pattern of the context vector files.  
-   `--split`: which split of data to work on (default: 1).
-   `--device`: computing device cuda or cpu.
-   `--iter-num`:  number of random tests to be done.
-   `--query`:  number of queries in test.
-   `--shot`: number of samples for each class in support set.
-   `--test-th`:  threshold for the test.
-   `--feat-dim`:  dimension of the output features from the encoder.
-   `--finetune-batch-size`:  batch size for training the STGCN.
-   `--finetune-epochs`:  epochs for training the STGCN.
-   `--finetune-lr`:  learning rate for training the STGCN.
-   `--selected-cls`: Selected test class for DoTA. None means testing on all classes.
-   `--ego-envolve`:  Specify use only ego or non-ego or both. Default test on ego.
-   `--save-model`: save the best model or not.
-   `--save-path`:  path to save the best model.
-   `--num-neighbors`:  num of neighbors in the STGCN model.
-   `--h-dim-1d`:  inter layer dim of the STGCN model.
-   `--gcn-groups`: number of groups in GCNeXt.
-   `--min-duration`: min duration of the input clips, should match the pre-computed context vectors.
-   `--max-duration`: max duration of the input clips, should match the pre-computed context vectors.
-   `--do-bk2`: stack an additional GCNeXt backbone.
-   `--fuse-type`:  method to integrate the node features to a graph feature ("mean" or "max").



## Results

Here is an example result:

```
selected_cls: None; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.8065, Std: 0.0821, Max acc: 1.0000, Min acc: 0.5333
Best model acc: normal 1.0000, abnormal 1.0000, overall 1.0000, train 1.0000

selected_cls: moving_ahead_or_waiting; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.8405, Std: 0.0823, Max acc: 1.0000, Min acc: 0.6333
Best model acc: normal 1.0000, abnormal 1.0000, overall 1.0000, train 1.0000

selected_cls: start_stop_or_stationary; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.7498, Std: 0.0770, Max acc: 0.9333, Min acc: 0.4333
Best model acc: normal 1.0000, abnormal 0.8667, overall 0.9333, train 1.0000

selected_cls: lateral; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.7917, Std: 0.0905, Max acc: 0.9667, Min acc: 0.5000
Best model acc: normal 1.0000, abnormal 0.9333, overall 0.9667, train 1.0000

selected_cls: turning; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.8265, Std: 0.0737, Max acc: 1.0000, Min acc: 0.6000
Best model acc: normal 1.0000, abnormal 1.0000, overall 1.0000, train 1.0000

selected_cls: oncoming; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.8287, Std: 0.0724, Max acc: 0.9667, Min acc: 0.6333
Best model acc: normal 1.0000, abnormal 0.9333, overall 0.9667, train 1.0000

selected_cls: pedestrian; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.7872, Std: 0.0837, Max acc: 0.9667, Min acc: 0.4667
Best model acc: normal 0.9333, abnormal 1.0000, overall 0.9667, train 1.0000

selected_cls: leave_to_left; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.8307, Std: 0.0674, Max acc: 0.9667, Min acc: 0.6000
Best model acc: normal 0.9333, abnormal 1.0000, overall 0.9667, train 1.0000

selected_cls: leave_to_right; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.7933, Std: 0.0852, Max acc: 0.9667, Min acc: 0.4667
Best model acc: normal 1.0000, abnormal 0.9333, overall 0.9667, train 1.0000

selected_cls: obstacle; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.8088, Std: 0.0721, Max acc: 0.9667, Min acc: 0.4000
Best model acc: normal 0.9333, abnormal 1.0000, overall 0.9667, train 1.0000

selected_cls: unknown; ego_envolve: True
2 way, 5 shot, 15 query, Num iters: 200
Mean acc: 0.7060, Std: 0.0854, Max acc: 0.9000, Min acc: 0.4333
Best model acc: normal 0.9333, abnormal 0.8667, overall 0.9000, train 1.0000
```

