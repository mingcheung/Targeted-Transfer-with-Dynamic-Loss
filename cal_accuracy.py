import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from utils import Normalize, load_ground_truth

model_1 = models.resnet50(pretrained=True).eval()
model_2 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()

for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

device = torch.device("cuda:0")
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(),])

filenames, true_labels, target_labels = load_ground_truth("dataset/images.csv")
n = len(filenames)

# Load images.
input_path = "dataset/images"
X = torch.zeros(n, 3, 299, 299).to(device)
for i in range(n):
    X[i] = trn(Image.open(os.path.join(input_path, filenames[i]+".png")))

# labels = torch.tensor(true_labels).to(device)
labels = torch.tensor(target_labels).to(device)

batch_size = 100
num_batches = np.int(np.ceil(n / batch_size))

preds_corr = 0
confs_corr = 0.
for i in range(num_batches):
    X_batch = X[i*batch_size: (i+1)*batch_size]
    labels_batch = labels[i*batch_size: (i+1)*batch_size]

    preds_batch = model_1(norm(X_batch))
    # preds_batch = (model_1(norm(X_adv_batch)) + model_2(norm(X_adv_batch)) + model_3(norm(X_adv_batch))) / 3
    idx_corr = torch.argmax(preds_batch, dim=1) == labels_batch
    preds_corr += sum(idx_corr)

    confs_batch = torch.softmax(preds_batch, dim=1)
    confs_batch = torch.max(confs_batch, dim=1).values
    confs_corr += sum(confs_batch[idx_corr])

accuracy = (preds_corr / len(labels)).cpu().numpy()
avg_conf = (confs_corr / preds_corr).cpu().numpy()
print("Accuracy/Attack success rate:", accuracy)
print("              Avg Confidence:", avg_conf)
