# Tongue shape classification based on IF-RCNet
by [Haowei Wang](https://github.com/WhwHcc)
### Framework:
![](https://github.com/WhwHcc/IFRCNet/blob/main/IFRCNet.png)
In the figure, the left part is the classification network RCNet, 
and the right side is the segmentation network RCUNet, 
and the two networks are nested to form IFRCNet.
## Usage:
### Requirement:
pytorch 2.0.1+python 3.10.13+ numpy 1.24.3
## To run:
You need to create three folders: "best_model", "data", "result", "data", 
which contain two files for the original image and mask map, 
and a txt file for the image address and tags, 
and "result" for the training results. 
You execute 'train' first and then 'test'.
### Public datasets:
the public Tongue dataset
link：https://github.com/BioHit/TongeImageDataset

## Note
* The repository is being updated
* Contact: Haowei Wang (15672806923@163.com)
