

"""

Evaluate simple transferabl attacks in the more challenging transfer setting
with no architecture overlap between the source ensemble models and the target model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm_notebook
import numpy as np
import scipy.stats as st
from utils import load_ground_truth, Normalize


##define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
channels=3
kernel_size=5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

##define DI
def DI(X_in):
    rnd = np.random.randint(299, 330,size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out 
    else:
        return  X_in


model_1 = models.resnet50(pretrained=True).eval()
model_2 = models.inception_v3(pretrained=True,transform_input=True).eval()
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

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(),])
image_id_list,label_ori_list,label_tar_list = load_ground_truth('./dataset/images.csv')


batch_size = 20
max_iterations = 300
input_path = './dataset/images/'
num_batches = np.int(np.ceil(len(image_id_list) / batch_size))
img_size = 299
lr = 2 / 255
epsilon = 16


# CE
# X_adv = []
pos = np.zeros((4,max_iterations // 20))
for k in tqdm_notebook(range(0,num_batches)):
    print("batch:", k)
    batch_size_cur = min(batch_size,len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur,3,img_size,img_size).to(device)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')
    for t in range(max_iterations):
        print("---iteration:", t)
        # logits = (model_1(norm(DI(X_ori + delta))) + model_2(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 4
        logits = (model_2(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 3
        # logits = (model_1(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 3
        # logits = (model_1(norm(DI(X_ori + delta))) + model_2(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 3
        # logits = (model_1(norm(DI(X_ori + delta))) + model_2(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta)))) / 3

        main_loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)

        alpha = 0.6
        n_topk = 5
        beta = np.arange(n_topk, 0, -1)
        beta = (1 - alpha) * beta / (np.sum(beta))
        labels_onehot = F.one_hot(labels, 1000)
        indices_topk = (logits + labels_onehot * (-9999999)).topk(n_topk)[1]
        aux_loss = torch.zeros(1, n_topk).to(device)
        for i in range(n_topk):
            aux_loss[0, i] = -nn.CrossEntropyLoss(reduction='sum')(logits, indices_topk[:, i])

        loss = alpha * main_loss + torch.mul(torch.tensor(beta).to(device), aux_loss).sum()
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
        grad_a = grad_c + 1 * grad_pre #MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori
        if t % 20 == 19:
            pos[0,t // 20] = pos[0,t // 20] + sum(torch.argmax(model_1(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[1,t // 20] = pos[1,t // 20] + sum(torch.argmax(model_2(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[2,t // 20] = pos[2,t // 20] + sum(torch.argmax(model_3(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[3,t // 20] = pos[3,t // 20] + sum(torch.argmax(model_4(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
    # for j in range(batch_size_cur):
    #     x_np = transforms.ToPILImage()((X_ori + delta)[j].detach().cpu())
    #     x_np.save(os.path.join(output_path,image_id_list[k * batch_size + j] + '.png'))
    # X_adv.append((X_ori + delta.data).cpu().numpy())
torch.cuda.empty_cache()
# X_adv = np.vstack(X_adv)
# np.save("exp_results/ensemble-based-attacks/X_adv_Inc-v3_Dense121_VGG16_M-TI-DI2-FGSM_dynamic-CE_alpha-"+str(alpha)+".npy", X_adv)
np.savetxt("exp_results/ensemble-based-attacks/succ_rates_Inc-v3_Dense121_VGG16_M-TI-DI2-FGSM_dynamic-CE_alpha-"+str(alpha)+".txt", pos/10, fmt="%.1f", delimiter=",")


# Logit
# X_adv = []
pos = np.zeros((4,max_iterations // 20))
for k in tqdm_notebook(range(0,num_batches)):
    print("batch:", k)
    batch_size_cur = min(batch_size,len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur,3,img_size,img_size).to(device)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')
    for t in range(max_iterations):
        print("---iteration:", t)
        # logits = (model_1(norm(DI(X_ori + delta))) + model_2(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 4
        logits = (model_2(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 3
        # logits = (model_1(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 3
        # logits = (model_1(norm(DI(X_ori + delta))) + model_2(norm(DI(X_ori + delta))) + model_4(norm(DI(X_ori + delta)))) / 3
        # logits = (model_1(norm(DI(X_ori + delta))) + model_2(norm(DI(X_ori + delta))) + model_3(norm(DI(X_ori + delta)))) / 3

        alpha = 0.6
        n_topk = 5
        beta = np.arange(n_topk, 0, -1)
        beta = (1 - alpha) * beta / (np.sum(beta))
        labels_onehot = F.one_hot(labels, 1000)
        logits_topk = (logits + labels_onehot * (-9999999)).topk(n_topk)[0]

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        logit_dists = ( -1 * real)

        loss = alpha*logit_dists.sum() + torch.mul(torch.tensor(beta).to(device), logits_topk).sum()
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
        grad_a = grad_c + 1 * grad_pre #MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori
        if t % 20 == 19:
            pos[0,t // 20] = pos[0,t // 20] + sum(torch.argmax(model_1(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[1,t // 20] = pos[1,t // 20] + sum(torch.argmax(model_2(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[2,t // 20] = pos[2,t // 20] + sum(torch.argmax(model_3(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[3,t // 20] = pos[3,t // 20] + sum(torch.argmax(model_4(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
    # for j in range(batch_size_cur):
    #     x_np = transforms.ToPILImage()((X_ori + delta)[j].detach().cpu())
    #     x_np.save(os.path.join(output_path,image_id_list[k * batch_size + j] + '.png'))
    # X_adv.append((X_ori + delta.data).cpu().numpy())
torch.cuda.empty_cache()
# X_adv = np.vstack(X_adv)
# np.save("exp_results/ensemble-based-attacks/X_adv_Inc-v3_Dense121_VGG16_M-TI-DI2-FGSM_dynamic-Logit_alpha-"+str(alpha)+".npy", X_adv)
np.savetxt("exp_results/ensemble-based-attacks/succ_rates_Inc-v3_Dense121_VGG16_M-TI-DI2-FGSM_dynamic-Logit_alpha-"+str(alpha)+".txt", pos/10, fmt="%.1f", delimiter=",")
