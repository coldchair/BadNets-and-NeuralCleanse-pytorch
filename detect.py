import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


import argparse
import os
import pathlib
import re
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_testset, build_clean_training_set
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet


parser = argparse.ArgumentParser(description='Neural Cleanse by PyTorch, detect trigger BadNets trained on MNIST.')


parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=1, help='Number of epochs to train backdoor model, default: 10')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to split dataset, default: 32')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.01')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cuda:0', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.3, help='poisoning portion (float, range from 0 to 1, default: 0.3)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

# 新加入
parser.add_argument('--trigger_label_plus', type = int, default = 3, help = 'The shift of label while poisoning')

parser.add_argument('--model_path', default=None, help='Path to load the model (default: ./checkpoints/badnet.pth)')

args = parser.parse_args()

device = None

def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((1, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        sum_losses = 0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {} loss : {}".format(norm, sum_losses))


        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu(), sum_losses

def main():
    # ----------------------------------- Init ----------------------------------- #
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

    dataset_train_clean = build_clean_training_set(is_train = True, args = args)
    data_loader_train_clean  = DataLoader(dataset_train_clean, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # for (x, y) in data_loader_train_clean:
    #     print(x.shape)


    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)

    basic_model_path = "./checkpoints/badnet-%s.pth" % args.dataset
    if (args.model_path != None):
        basic_model_path = args.model_path
    model.load_state_dict(torch.load(basic_model_path), strict=True)

    # ----------------------------- reverse_engineer ----------------------------- #
    param = {
        "dataset": "MNIST",
        "Epochs": 5,
        "batch_size": 32,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (28, 28)
    }

    train_loader = data_loader_train_clean

    import numpy as np
    def compute_anomaly_index(losses):
        median = np.median(losses)
        mad = np.median(np.abs(losses - median))
        anomaly_scores = (losses - median) / (mad + 1e-6)
        return anomaly_scores

    norm_list = []
    losses = []

    for label in range(param["num_classes"]):
    # for label in range(7, 8):
        trigger, mask, loss = train(model, label, train_loader, param)
        norm_list.append(mask.sum().item())
        losses.append(loss)

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        plt.imshow(trigger, cmap='gray')
        plt.savefig('mask/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig('mask/mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)
    
    a_index = compute_anomaly_index(losses)
    print('a-index')
    print(a_index)

    print('norm')
    print(norm_list)



















if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()