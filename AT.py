import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
import numpy as np
import attack_generator as attack
from utils import Logger
from datetime import datetime
from models.modules import *
from models.ResNet_CIFAR import *

# Ensure the CUDA devices are properly set
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Argument parser for hyperparameters and configurations
parser = argparse.ArgumentParser(description='Adversarial Training (AT)')
parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation bound for adversarial attack')
parser.add_argument('--num-steps', type=int, default=10, help='number of perturbation steps for attack')
parser.add_argument('--step-size', type=float, default=2/255, help='step size for perturbation')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet20", help="Network to use: resnet20, preactresnet18, WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="Dataset: cifar10, cifar100")
parser.add_argument('--random', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width-factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop-rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--resume', type=str, default=None, help='path to resume training from checkpoint')
parser.add_argument('--out-dir', type=str, default='./checkpoints/train_dl_resnet20_cifar10', help='output directory')
parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'], help="Learning rate schedule type")
parser.add_argument('--lr-max', default=0.001, type=float, help='maximum learning rate')
parser.add_argument('--lr-one-drop', default=0.01, type=float, help='learning rate after one drop')
parser.add_argument('--lr-drop-epoch', default=100, type=int, help='epoch at which to drop learning rate')
args = parser.parse_args()

# Training setup with seeds and configurations
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True  # Allow the backend to optimize performance based on hardware
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior

# Model and optimizer setup
if args.net == "resnet20":
    model = resnet20("full").cuda()
    net = "resnet20"
# Add other network options (e.g., preactresnet18, WRN) if needed

# Load pretrained model if fine-tuning
pretrained_dict = torch.load(args.resume, map_location=torch.device('cpu')) if args.resume else None
if pretrained_dict:
    model.load_state_dict(pretrained_dict)

optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

# Learning rate scheduling options
def get_lr_schedule():
    if args.lr_schedule == 'superconverge':
        return lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
        return lr_schedule
    elif args.lr_schedule == 'linear':
        return lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            return args.lr_max if t < args.lr_drop_epoch else args.lr_one_drop
        return lr_schedule
    elif args.lr_schedule == 'multipledecay':
        return lambda t: args.lr_max - (t // (args.epochs // 10)) * (args.lr_max / 10)
    elif args.lr_schedule == 'cosine': 
        return lambda t: args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

lr_schedule = get_lr_schedule()

# Make output directory if it doesn't exist
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# Save checkpoint function
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.out_dir, filename))

# Training loop for adversarial training
def train(epoch, model, train_loader, optimizer):
    model.train()
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # Generate adversarial examples using PGD attack
        x_adv, _ = attack.PGD(model, data, target, args.epsilon, args.step_size, args.num_steps, loss_fn="cent", category="Madry", rand_init=args.random)

        optimizer.param_groups[0].update(lr=lr_schedule(epoch + 1))
        optimizer.zero_grad()

        logit = model(x_adv)  # Forward pass on adversarial examples
        loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)  # Compute loss
        
        train_robust_loss += loss.item() * len(x_adv)
        
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights
        
        num_data += len(data)

    return train_robust_loss / num_data

# Setup data loader for CIFAR-10 or CIFAR-100 based on user input
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../cifar10', train=False, download=True, transform=transform_test)
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../cifar100', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../cifar100', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Resume from checkpoint if specified
start_epoch = 0
best_acc = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['test_pgd10_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title='AT', resume=True)
else:
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title='AT')
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD10 Acc'])

# Training loop starts here
for epoch in range(start_epoch, args.epochs):
    print(f"Epoch {epoch + 1}/{args.epochs} - {datetime.now()}")
    
    train_robust_loss = train(epoch, model, train_loader, optimizer)

    # Evaluate the model's performance on clean and adversarial data
    _, test_nat_acc = attack.eval_clean(model, test_loader)
    _, test_pgd10_acc = attack.eval_robust(model, test_loader, perturb_steps=10, epsilon=8/255, step_size=2/255, loss_fn="cent", category="Madry", random=True)

    print(f"Epoch: {epoch+1}/{args.epochs} | LR: {lr_schedule(epoch+1)} | Natural Test Acc: {test_nat_acc:.2f} | PGD10 Test Acc: {test_pgd10_acc:.2f}")

    logger_test.append([epoch + 1, test_nat_acc, test_pgd10_acc])

    # Save the best checkpoint based on PGD10 accuracy
    if test_pgd10_acc > best_acc:
        best_acc = test_pgd10_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd10_acc': test_pgd10_acc,
            'optimizer': optimizer.state_dict(),
        }, filename='finetune_bestpoint.pth.tar')

    # Save the last checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'test_nat_acc': test_nat_acc,
        'test_pgd10_acc': test_pgd10_acc,
        'optimizer': optimizer.state_dict(),
    })

# Close logger at the end
logger_test.close()
