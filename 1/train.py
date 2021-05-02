import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
				default=False,
				help='whether download the dataset or not',
				type=bool)
ap.add_argument('--resume',
				default=None,
				help='path of pretrained model',
				type=str)

args = ap.parse_args()
data_name = args.data_name.lower()
input_root = args.input_root
output_root = args.output_root
num_epoch = args.num_epoch
download = args.download
resume = args.resume


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


lr = 0.001 #学习率
momentum = 0.5
batch_size = 128


print("Preparing data...")
data_transform = transforms.Compose([
	#transforms.Resize(),
    transforms.ToTensor(),
#    normalize
])
train_data_transform = transforms.Compose([
	#transforms.Resize(),
	#transforms.RandomRotation(90),
	transforms.ToTensor(),
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


print("Training model...")
def train(model, train_loader, task, device, epoch, optimizer):
	''' One epoch training
	return loss
	'''
	model.train()
	train_loss = 0
	y_true = torch.tensor([]).to(device)
	y_score = torch.tensor([]).to(device)
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		data, target = Variable(data), Variable(target)
		
		optimizer.zero_grad()	
		output = model(data)
		#output = nn.functional.softmax(output,dim=1)
		if task == 'multi-label, binary-class':
			loss = F.binary_cross_entropy_with_logits(output, target.float())
		else:
			loss = F.cross_entropy(output, target.long().squeeze())
			target = target.float().resize_(len(target), 1)
		
		y_true = torch.cat((y_true, target))
		y_score = torch.cat((y_score, nn.functional.softmax(output, dim=1)))

		train_loss += loss.item()

		loss.backward()
		optimizer.step()

	y_true = y_true.cpu().numpy()
	y_score = y_score.detach().cpu().numpy()
	acc = getACC(y_true, y_score, task)
	train_loss /= len(train_loader)
	return train_loss, acc

def val(model, val_loader, task, device):
	model.eval()

	val_loss = 0
	y_true = torch.tensor([]).to(device)
	y_score = torch.tensor([]).to(device)
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(val_loader):
			data = data.to(device)
			target = target.to(device)
		
			output = model(data)
			output = nn.functional.softmax(output,dim=1)

			if task == 'multi-label, binary-class':
				loss = F.binary_cross_entropy_with_logits(output, target.float())
			else:
				loss = F.cross_entropy(output, target.long().squeeze())
				target = target.float().resize_(len(target), 1)
			val_loss += loss.item()
			y_true = torch.cat((y_true, target))
			y_score = torch.cat((y_score, F.softmax(output, dim=1)))


	y_true = y_true.cpu().numpy()
	y_score = y_score.cpu().numpy()
	auc = getAUC(y_true, y_score, task)
	acc = getACC(y_true, y_score, task)
	val_loss /= len(val_loader)
	return val_loss, auc, acc

def test(model, test_loader, task, device):
    model.eval()  
    test_loss = 0

    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
    	for batch_idx, (data, target) in enumerate(test_loader):
 
        	data = data.to(device)
        	target = target.to(device)
        	output = model(data)
        	#output = nn.functional.softmax(output,dim=1)
        	if task == 'multi-label, binary-class':
        		loss = F.binary_cross_entropy_with_logits(output, target.float())
        	else:
        		loss = F.cross_entropy(output, target.long().squeeze())
        		target = target.float().resize_(len(target), 1)
        	test_loss += loss.item()
        	y_true = torch.cat((y_true, target))
        	y_score = torch.cat((y_score, F.softmax(output, dim=1)))

    y_true = y_true.cpu().numpy()
    y_score = y_score.cpu().numpy()
    auc = getAUC(y_true, y_score, task)
    acc = getACC(y_true, y_score, task)
    test_loss /= len(test_loader)  

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}/{}\n'.format(
        test_loss, acc, len(test_loader.dataset)))

    return test_loss, auc, acc

dir_path = os.path.join(output_root, '%s_checkpoints' % (data_name))
if not os.path.exists(dir_path):
	os.makedirs(dir_path)
train_loss = []
train_acc = []
val_loss = []
val_auc = []
val_acc = []
start_epoch = 0
if resume:
	checkpoint = torch.load(resume)
	model.load_state_dict(checkpoint['net'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	start_epoch = checkpoint['epoch']

for epoch in range(start_epoch+1, num_epoch+1):
	trainl, trainacc = train(model, train_loader, task, device, epoch, optimizer)

	print("Train Epoch : {}/{:.0f}\t Loss:{:.4f}\t Accuracy:{:.4f}".format(epoch, num_epoch, trainl, trainacc))
	checkpoint = {
	'net': model.state_dict(),
	'optimizer': optimizer.state_dict(),
	'epoch':epoch
	}
	trained_path = os.path.join(dir_path,'ckpt_%s.pth'%(str(epoch)))
	torch.save(checkpoint, trained_path)

	print("State Saved")
	train_loss.append(trainl)
	train_acc.append(trainacc)
	vall, valauc, valacc = val(model, val_loader, task, device)
	val_loss.append(vall)
	val_auc.append(valauc)
	val_acc.append(valacc)

testl, testauc, testacc = test(model, test_loader, task, device)

#torch.save(model, output_root+'/classifier.pkl')

N = np.arange(1, num_epoch-start_epoch+1)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, train_loss, label="train_loss")
plt.plot(N, train_acc, label="train_acc")
plt.plot(N, val_loss, label="val_loss")
plt.plot(N, val_acc, label="val_acc")
plt.title("Training Loss and Accuracy on %s (%s)" % (data_name, args.model))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(os.path.join(output_root, "%s_result.jpg" % (data_name)))
plt.show()
	

