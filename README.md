ResNet training in Torch
============================

Visual experiments of the paper: [Multi-label Music Genre Classification from Audio, Text, and Images Using Deep Features](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/126_Paper.pdf). [Repo](https://github.com/sergiooramas/tartarus#ismir-2017-experiments-multi-label-classification) of audio and textual experiments.

This is a modified version of [Facebook resnet (torch)](https://github.com/facebook/fb.resnet.torch).

Requirments and installation [installation instructions](INSTALL.md).

See the [training recipes](TRAINING.md) for complete examples.

## Finetuning on our dataset

Download the [dataset](https://zenodo.org/record/831189#.WlzidXWnFB0). Note that you have to organize the files as follow:
* train/
  * label1/
  * ...
  * labeln/
* test/
  * label1/
  * ...
  * labeln/
* val/
  * label1/
  * ...
  * labeln/

Download the [Imagnet pretrained model](https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7) (resnet-101).

To finetune a resnet-101 pretrained model on the dataset run:
```bash
th main.lua -save <PATH_TO_NEW_MODEL> -LR 0.0001 -batchSize 50 -retrain <PATH_TO_PRETRAINED_MODEL> -data <PATH_TO_DATA> -resetClassifier true -nClasses 250
```

For example:
```bash
th main.lua -save checkpoints/ -LR 0.0001 -batchSize 50 -retrain misc/resnet-101.t7 -data misc/dataset/ -resetClassifier true -nClasses 250
```

## Extract visual features
Download our resnet [model](https://drive.google.com/open?id=1LWXVVBYSraFYsqepbsl_OkeZ8pSLMPaC) (model_best.t7).

```bash
th extract-features.lua misc/model_best.t7 30 <IMAGES_LIST>
```

it will be saved in the main directory a numpy array N x D, where N is the number of files and D the dimensions of the vectors (2048). 
<IMAGES_LIST> is a text file containing the list of N images (full path) that you want to process.