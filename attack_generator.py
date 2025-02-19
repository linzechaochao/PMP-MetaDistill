import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Custom loss function for Carlini-Wagner (CW) attack
def cwloss(output, target, confidence=50, num_classes=10):
    # One-hot encoding of the target
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,)).cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    # Compute the loss for the CW attack
    real = (target_var * output).sum(1)  # Real class score
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]  # Maximum score of the other classes
    loss = -torch.clamp(real - other + confidence, min=0.)  # Ensure the loss is non-negative
    return torch.sum(loss)

# Projected Gradient Descent (PGD) attack
def PGD(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()  # Set model to evaluation mode to avoid dropout or batch norm updates
    Kappa = torch.zeros(len(data))  # To track the number of successful adversarial perturbations

    # Initialize adversarial examples (random noise for TRADES, uniform noise for Madry)
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)  # Natural output (non-adversarial)
    elif category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Ensure the adversarial example stays within valid bounds

    # Perform the PGD attack for the specified number of steps
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]  # Predicted class

        # Track how many times the adversarial example is classified correctly
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        
        model.zero_grad()
        with torch.enable_grad():
            # Compute the adversarial loss
            if loss_fn == "cent":
                loss_adv = F.cross_entropy(output, target)
            elif loss_fn == "cw":
                loss_adv = cwloss(output, target)
            elif loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_output, dim=1))
        
        loss_adv.backward()  # Compute gradients with respect to the adversarial example
        eta = step_size * x_adv.grad.sign()  # Update rule for PGD (sign of the gradient)
        
        # Apply the perturbation and ensure it's within epsilon bounds
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Ensure the adversarial example is within valid image range

    # Return the adversarial example and the count of successful attacks (Kappa)
    return Variable(x_adv, requires_grad=False), Kappa

# Evaluate the model on clean (non-adversarial) data
def eval_clean(model, test_loader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # No need to compute gradients for evaluation
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)  # Average loss
    test_accuracy = correct / len(test_loader.dataset)  # Accuracy as fraction of correct predictions
    return test_loss, test_accuracy

# Evaluate the model on adversarial examples generated via PGD
def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.enable_grad():  # Enable gradient computation for the adversarial examples
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv, _ = PGD(model, data, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=random)
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # Average loss
    test_accuracy = correct / len(test_loader.dataset)  # Accuracy on adversarially perturbed data
    return test_loss, test_accuracy
