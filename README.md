# CSE547FinalProject
This is the repository for the final project of CSE547. In this project, I did some experiments for the crowd counting task. 

# Dataset
In this project, three different models are tested on the ShanghaiTech datasets. The datasets can be downloaded from here: [Google Drive](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view).

# Platform to run the codes
1. Before running the codes, please make sure you have install the Pytorch deep learning framework. For more details, please check here: [PyTorch](https://pytorch.org/). 
2. I trained and tested my codes under Linux 18.04 with a Nvidia Tesla GPU v100. 
3. Before running the codes, you need to generate your own *.json* files.
For example, if you entering the *CSRNet* folder, please open and edit the file *part_A_train_with_val_240.json*. You need to change the file path to your own.
```python
For example:
/YOUR/OWN/PATH/Datasets/ShanghaiTech/part_A_final/train_data/images/IMG_300.jpg
```
# How to run the codes
There are three different models: MCNN, CSRNet, and modified-CSRNet. For example, if you want to train the CSRNet.
1. Enter the *CSRNet* folder;
2. type the following command:

```python
$ python train.py part_A_train_with_val_240.json part_A_val_59.json 0 part_A_
```
You can use the similar command to train the ShanghaiTech Part_B dataset. Moreover, for the rest two models (i.e., MCNN and modified-CSRNet), please follow the same steps. 

# How to test the models and generate density maps
Suppose you are located in the folder of CSRNet, to test the model, please type the following command:

```python
$ python testMaeMse.py
```
Some simple modifications probabaly are needed (such as changing paths) before generating the density maps. 
Also, to generate the density maps, you need to use your own saved models. However, I also provide the saved models I generated during experiments. Here is the link: [Saved Models](https://drive.google.com/drive/folders/1SESnPh4XmXjnowlqZXgTo3zlo98WwlIG?usp=sharing). 

# Samples of generated density maps
Several generated density maps of the ShanghaiTech Part_A dataset have been uploaded to the folder *Generated Maps*.

# Contact information
If you have any question, please contact me by email (zuhui.wang@stonybrook.edu).
