## About
PyTorch code for the paper: **Dynamic Loss Yielding More Transferable Targeted Adversarial Examples**.


### Requirements
torch>=1.7.0; torchvision>=0.8.1; tqdm>=4.31.1; pillow>=7.0.0; matplotlib>=3.2.2;  numpy>=1.18.1; 

### Dataset
The 1000 images from the NIPS 2017 ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv``` and ```dataset/imagenet_class_index.json```.

### Evaluation
```eval_single.py```: Generate targeted adversarial examples on a single model with the traditional loss. 

```eval_single_dyn_loss.py```: Generate targeted adversarial examples on a single model with the dynamic loss. 

```eval_ensemble.py```: Generate targeted adversarial examples on an ensemble model with the traditional loss. 
 
```eval_ensemble_dyn_loss.py```: Generate targeted adversarial examples on an ensemble model with the dynamic loss. 

```cal_accuracy```: Calculate the accuracy of adversarial or normal examples. 

```utils.py```: Some necessary utility functions.