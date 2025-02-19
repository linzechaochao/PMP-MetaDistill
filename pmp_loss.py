import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

# PGD attack function
def attack_pgd(model, train_batch_data, train_batch_labels, attack_iters=10, step_size=2/255.0, epsilon=8.0/255.0):
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    # Initialize adversarial examples with random noise within epsilon bounds
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon, epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data, 0, 1)  # Ensure within image bounds [0, 1]
    
    # Perform PGD attack iteratively
    for _ in range(attack_iters):
        train_ifgsm_data.requires_grad_()  # Enable gradient computation for adversarial examples
        logits = model(train_ifgsm_data)  # Forward pass
        loss = ce_loss(logits, train_batch_labels.cuda())  # Compute cross-entropy loss
        loss.backward()  # Backpropagate to compute gradients
        train_grad = train_ifgsm_data.grad.detach()  # Get the gradient
        train_ifgsm_data = train_ifgsm_data + step_size * torch.sign(train_grad)  # Apply gradient update
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(), 0, 1)  # Ensure adversarial example is within valid image range

        # Apply perturbation clipping to ensure it remains within epsilon bounds
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert, -epsilon, epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert  # Add the perturbation to original data
        train_ifgsm_data = train_ifgsm_data.detach()  # Detach to prevent gradient flow
        
    return train_ifgsm_data

# PMP inner loss function with cross-entropy loss for adversarial training
def pmp_inner_loss_ce(model, teacher_adv_model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0):
    criterion_ce_loss = torch.nn.CrossEntropyLoss().cuda()  # CrossEntropy loss criterion
    
    # Set model to evaluation mode for generating adversarial examples
    model.eval()
    
    # Initialize adversarial examples with small random noise
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    # Generate adversarial examples using PGD attack
    for _ in range(perturb_steps):
        x_adv.requires_grad_()  # Enable gradient computation for adversarial examples
        with torch.enable_grad():
            # Compute the cross-entropy loss for adversarial examples
            loss_ce = criterion_ce_loss(model(x_adv), y.cuda())
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]  # Get the gradient with respect to the adversarial example
        # Update adversarial example using the sign of the gradient
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)  # Clip within epsilon bounds
        x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Ensure the perturbation is valid within the image range

    model.train()  # Set the model back to training mode

    # Detach adversarial examples to prevent gradient flow during feature extraction
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    # Zero the gradients before performing optimization step
    optimizer.zero_grad()

    # Get the student's logits and adversarial features
    student_logits = model(x_adv)
    student_adv_feature_list = model.feature_list

    # With no gradient computation for teacher model, get teacher's logits and features
    with torch.no_grad():
        teacher_logits = teacher_adv_model(x_adv)
        teacher_adv_feature_list = teacher_adv_model.feature_list
        
    # Return necessary logits and features for PMP loss computation
    return student_logits, teacher_logits, teacher_adv_feature_list, student_adv_feature_list
