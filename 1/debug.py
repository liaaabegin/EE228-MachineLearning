import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data

import medmnist.models as models
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model',
				default='ResNet18',
				help='training model, ResNet18 or ResNet50',
				type=str)
ap.add_argument('--data_name',
				default='pathmnist',
				help='subset of MedMNIST',
				type=str)
ap.add_argument('--input_root',
				default='./input',
				help='input root, the source of dataset files',
				type=str)
ap.add_argument('--output_root',
				default='./output',
				help='output root, where to save models and results',
				type=str)
ap.add_argument('--num_epoch',
				default=100,
				help='num of epochs of training',
				type=int)
ap.add_argument('--download',
				default=True,
				help='whether download the dataset or not',
				type=bool)

args = ap.parse_args()
data_name = args.data_name.lower()
input_root = args.input_root
output_root = args.output_root
num_epoch = args.num_epoch
download = args.download

flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
DataClass = flag_to_class[data_name]

info = INFO[data_name]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
model = getattr(models, args.model)(in_channels=n_channels, num_classes=n_classes)

lr = 0.01 #学习率
momentum = 0.5
#log_interval = 20 #跑多少次batch进行一次日志记录
batch_size = 128
dir_path = os.path.join(output_root, '%s_checkpoints' % (data_name))
if not os.path.exists(dir_path):
	os.makedirs(dir_path)


print("Preparing data...")
data_transform = transforms.Compose([
    transforms.ToTensor(),
#    normalize
])
train_data_transform = transforms.Compose([
	transforms.RandomRotation(90),
	transforms.ToTensor(),
#	normalize
	])
train_data =  DataClass(root=input_root,
						split='train',
						transform=train_data_transform,
						download=download)
val_data = DataClass(root=input_root,
					split='val',
					transform=data_transform,
					download=download)
test_data =  DataClass(root=input_root,
						split='test',
						transform=data_transform,
						download=download)
train_loader = data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = data.DataLoader(
	val_data,
	batch_size=batch_size,
	shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train(model, train_loader, task, device, epoch, optimizer):
	''' One epoch training
	return loss
	'''
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(device)
		target = target.to(device)
		data, target = Variable(data), Variable(target)
		
		optimizer.zero_grad()	
		output = model(data)

		if task == 'multi-class':
			loss = F.cross_entropy(output, target.type(torch.long).squeeze())
		else:
			loss = F.binary_cross_entropy_with_logits(output, target)
		loss.backward()
		optimizer.step()

	return loss.item()

def val(model, val_loader, task, device):
	model.eval()

	y_true = torch.tensor([]).to(device)
	y_score = torch.tensor([]).to(device)
	for batch_idx, (data, target) in enumerate(val_loader):
		data = data.to(device)
		target = target.to(device)
		data, target = Variable(data), Variable(target)
		
		output = model(data)
		model.train()

		if task == 'multi-class':
			target = target.type(torch.long).squeeze()
			loss = F.cross_entropy(output, target).item()
			target = target.float().resize_(len(target), 1)
		else:
			loss = F.binary_cross_entropy_with_logits(output, target).item()

		y_true = torch.cat(y_true, target, 0)
		y_score = torch.cat(y_score, output, 0)

	y_true = y_true.cpu().numpy()
	y_score = y_score.cpu().numpy()
	auc = getAUC(y_true, y_score, task)
	acc = getACC(y_true, y_score, task)

	return loss, auc, acc

y_true = torch.tensor([]).to(device)
y_score = torch.tensor([]).to(device)

test_loss = 0
for idx, (data, target) in enumerate(train_loader):
	model.train()
	data = data.to(device)
	target = target.to(device)
	data, target = Variable(data), Variable(target)		

	output = model(data)
	print(len(target))
	for i in range(5):
		print("target: ", end='')
		print(target[i])
		#print("target: ", end='')
		#print(target[i].long().squeeze())
		print("output: ", end='')
		print(output[i])
		print("Softmax(output): ", end='')
		print(F.softmax(output[i], dim=0))
		print("MySoftmax(output): ", end='')
		print(torch.exp(output[i])/torch.sum(torch.exp(output[i])))
	#if task == 'multi-label, binary-class':
	#	loss = F.binary_cross_entropy_with_logits(output, target.squeeze().float())
	#else:
	#	loss = nn.CrossEntropyLoss()
	#	loss = loss(output, target)
	
	#test_loss += loss.item()
	#loss.backward()
	#optimizer.step()

	break

