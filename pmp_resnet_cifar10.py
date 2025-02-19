import os
import argparse
import torch
import torch.optim as optim
from pmp_loss import *
import torchvision
from torchvision import datasets, transforms
from models.ResNet_CIFAR_feature_align import *
from models import resnet_normal_DL
from datetime import datetime
from models.meta_util import LogitsWeight
from models.meta_optimizer import MetaSGD
from models import DistillKL
import json
from functions import HardBuffer
import torchattacks

# Set deterministic behavior for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# CUDA device setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Hyperparameters and other configurations
prefix = 'resnet_CIFAR10_pmp_'
epochs = 200
batch_size = 128
epsilon = 8/255.0
weight_learn_rate = 0.025
lr_decay_rate = 0.1
lr_decay_epochs = [125, 150, 175]
bert = 1
adv_teacher_path = ''
nat_teacher_path = ''
cifar10_path = ''

file_path = 'meta_logit_loss_objective_cx_rx.json'

# Data augmentation and preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize student model and optimizer
student = resnet20('full').cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

# Initialize the distillation loss functions and weight parameters
criterion_div = DistillKL(4)
weight = {
    "adv_loss": 1/2.0,
    "nat_loss": 1/2.0,
}
init_loss_nat = None
init_loss_adv = None

# Load the teacher models
teacher = resnet_normal_DL.resnet110()
teacher.load_state_dict(torch.load(adv_teacher_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
teacher = teacher.cuda()
teacher.eval()

teacher_nat = resnet56('full')
teacher_nat.load_state_dict(torch.load(nat_teacher_path, map_location=torch.device('cpu'))['state_dict'])
teacher_nat = teacher_nat.cuda()
teacher_nat.eval()

# Loss functions
KL_loss = nn.KLDivLoss(reduction='batchmean')
MSE_loss = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()

# Setup for weight optimization
n_cls = 10
teacher_num = 2
learning_rate = 0.05
momentum = 0.9
weight_decay = 5e-4
rollback = True
nesterov = True
WeightLogits = LogitsWeight(n_feature=n_cls*(teacher_num+2), teacher_num=teacher_num).cuda()
weight_params = list(WeightLogits.parameters())
weight_optimizer = optim.Adam(weight_params, lr=1e-3, weight_decay=1e-4)
model_s_params = list(student.parameters())

# Setup for meta-optimizer
model_s_optimizer = MetaSGD(model_s_params,
                             [student],
                             lr=learning_rate,
                             momentum=momentum,
                             weight_decay=weight_decay, 
                             nesterov=nesterov, 
                             rollback=rollback, 
                             cpu=False)

# Hard buffer initialization
hard_buffer = True
cal_json = []
buffer_size = 512
if hard_buffer:
    hardBuffer = HardBuffer(batch_size=batch_size, buffer_size=buffer_size)

# Function to calculate the loss for each epoch
def cal_epoc_loss_soft_c_r_w(softmax_list):
    epoch_loss = 0.0
    sft_ele_c_l = 0.0
    sft_ele_r_l = 0.0
    for ele in softmax_list:
        sfm_ele, loss_ele = ele[0], ele[1]
        sft_ele_c, sft_ele_r = sfm_ele[0], sfm_ele[1]
        sft_ele_c_l += sft_ele_c
        sft_ele_r_l += sft_ele_r
        epoch_loss += loss_ele

    epoch_loss = epoch_loss / len(softmax_list)
    sft_ele_c_l = sft_ele_c_l / len(softmax_list)
    sft_ele_r_l = sft_ele_r_l / len(softmax_list)

    return epoch_loss, sft_ele_c_l, sft_ele_r_l

# Function to write the results to a JSON file
def write_to_json(total_list):
    # Convert the list to a JSON string
    json_data = json.dumps(total_list)

    # Save the JSON string to a file
    with open(file_path, 'w') as json_file:
        json_file.write(json_data)

    print(f'List has been successfully saved to {file_path}')

# Inner objective function for the PMP loss
def inner_objective(train_batch_data, train_batch_labels, teacher_nat, student, teacher, matching_only=False):
    with torch.no_grad():
        teacher_nat_logits = teacher_nat(train_batch_data)
        teacher_nat_feature_list = teacher_nat.feature_list

    student_adv_logits, teacher_adv_logits, teacher_adv_feature_list, student_adv_feature_list = pmp_inner_loss_ce(
        student, teacher, train_batch_data, train_batch_labels, optimizer, step_size=2/255.0, epsilon=epsilon, perturb_steps=10
    )
    student.train()

    student_nat_logits = student(train_batch_data)
    student_nat_feature_list = student.feature_list

    logit_t_list = [teacher_nat_logits, teacher_adv_logits]
    logit_s_list = [student_nat_logits.detach(), student_adv_logits.detach()]

    loss_div_list_nat = [criterion_div(student_nat_logits.detach(), logit_t, is_ca=True)
                         for logit_t in logit_t_list]

    loss_div_list_adv = [criterion_div(teacher_adv_logits.detach(), logit_t, is_ca=True)
                         for logit_t in logit_t_list]

    loss_div_list = list(np.add(loss_div_list_nat, loss_div_list_adv) * 0.5)
    loss_div = torch.stack(loss_div_list, dim=1)
    logits_weight = WeightLogits(logit_t_list, logit_s_list)
    
    loss_div = torch.mul(logits_weight, loss_div).sum(-1).mean()

    loss_feature_nat = torch.FloatTensor([0.]).cuda()
    loss_feature_adv = torch.FloatTensor([0.]).cuda()
    loss_feature = torch.FloatTensor([0.]).cuda()

    # Calculate feature map losses
    for index in range(len(student_adv_feature_list)):
        if index == 0:
            loss_feature_adv += MSE_loss(student_adv_feature_list[index], teacher_adv_feature_list[index])
        else:
            loss_feature_nat += MSE_loss(student_nat_feature_list[index], teacher_nat_feature_list[index])
    loss_feature = loss_feature_nat + loss_feature_adv

    if matching_only:
        total_loss = loss_div * 0.5 + 0.5 * loss_feature
        return total_loss, logits_weight

    loss_cls = criterion_cls(student_nat_logits, train_batch_labels)
    total_loss = 0.25 * loss_cls + 0.25 * loss_div + 0.5 * loss_feature

    if hard_buffer:
        if not hardBuffer.is_full():
            hardBuffer.put(train_batch_data, train_batch_labels)
        else:
            bo = (student_nat_logits.argmax(1) != train_batch_labels)
            hardBuffer.update(train_batch_data[bo], train_batch_labels[bo])

    return total_loss, logits_weight

# Outer objective function to calculate classification loss
def outer_objective(student, data, criterion_cls):
    train_batch_data, target = data
    train_batch_data = train_batch_data.float()

    student_nat_logits = student(train_batch_data)
    loss_cls_clean = criterion_cls(student_nat_logits, target)

    # Generate adversarial examples using PGD attack
    attack = torchattacks.PGD(student, eps=8/255, alpha=2/255, steps=10, random_start=True)
    adv_samples = attack(train_batch_data, target)
    student_adv_logits = student(adv_samples)
    loss_cls_adv = criterion_cls(student_adv_logits, target)

    loss_cls = 0.5 * (loss_cls_clean + loss_cls_adv)
    return loss_cls, 0.0 

# Function to adjust learning rate during training
def adjust_learning_rate_cifar(optimizer, epoch):
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

# Training loop
for epoch in range(1, epochs + 1):
    print(datetime.now())
    adjust_learning_rate_cifar(model_s_optimizer, epoch)

    softmax_list = []
    epoch_loss_list = []

    print(f'The {epoch}th epoch')
    for step, (train_batch_data, train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        model_s_optimizer.zero_grad()
        total_loss = torch.tensor(0., requires_grad=True)
        total_loss, logits_weight = inner_objective(train_batch_data, train_batch_labels, teacher_nat, student, teacher)
        
        total_loss.backward()
        model_s_optimizer.step(None)

        # Perform operations related to hard buffer if enabled
        if hard_buffer:
            if not hardBuffer.is_full():
                hardBuffer.put(train_batch_data, train_batch_labels)
            else:
                bo = (student_nat_logits.argmax(1) != train_batch_labels)
                hardBuffer.update(train_batch_data[bo], train_batch_labels[bo])

        logits_weight = torch.sum(logits_weight, dim=0, keepdim=True) / logits_weight.shape[0]
        softmax_list.append([logits_weight.tolist()[0], total_loss.item()])

    epoch_loss, sft_ele_c_l, sft_ele_r_l = cal_epoc_loss_soft_c_r_w(softmax_list)
    cal_json.append([epoch, epoch_loss, sft_ele_c_l, sft_ele_r_l])
    print(epoch, epoch_loss, sft_ele_c_l, sft_ele_r_l)

# Optionally save results to JSON after training
# write_to_json(cal_json)
