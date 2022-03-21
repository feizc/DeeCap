# DeeCap

This repository includes the reference code for paper: Dynamic Early Exit for Efficient Image Captioning 


## Data 

To run the code, annotations and images for the COCO dataset are needed.
Please download the zip files including the images ([train2014.zip](http://images.cocodataset.org/zips/train2014.zip), [val2014.zip](http://images.cocodataset.org/zips/val2014.zip)),
the zip file containing the annotations ([annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)) and extract them. These paths will be set as arguments later. 
Our code supports the image features extracted from conventional Faster-RCNN or [CLIP](https://github.com/openai/CLIP) model.


## Training Procedure 

Run `python train_deecap.py` using the following arguments: 

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |



## Evaluation

To reproduce the results reported in our paper, download the checkpoint model file and place it in the ckpt folder.

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations | 


## Acknowledgment
This repository refers to [Transformer Image Captioning](https://github.com/aimagelab/meshed-memory-transformer) and [huggingface DeeBERT](https://github.com/huggingface/transformers/tree/master/examples/research_projects/deebert).



