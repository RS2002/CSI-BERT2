from model import CSIBERT,Sequence_Classifier
from transformers import BertConfig,AdamW
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data_random,load_data
import copy
from torch.utils.data import ConcatDataset

pad=-1000

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hs', type=int, default=128)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--intermediate_size', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--position_embedding_type', type=str, default="absolute")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0], help="CUDA device ids")
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--data_path', type=str, default="./data/data_sequence.pkl")
    parser.add_argument('--magnitude_path', type=str, default="recover.npy")
    parser.add_argument('--parameter', type=str, default=None)
    parser.add_argument('--train_prop', type=float, default=0.9)

    parser.add_argument('--class_num', type=int, default=6) #action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"
    parser.add_argument("--path", type=str, default='./csibert_pretrain.pth')
    parser.add_argument("--no_pretrain", action="store_true",default=False)
    parser.add_argument("--mask", action="store_true",default=False)
    parser.add_argument("--random_input", action="store_true",default=False)
    parser.add_argument("--recover_data", action="store_true",default=False)
    parser.add_argument("--freeze", action="store_true",default=False)
    parser.add_argument("--dataset", type=str, default="WiGesture")

    parser.add_argument('--mode', type=int, default=0) # 0: train(100Hz),test(100Hz); 1: train(100Hz+50Hz),test(100Hz+50Hz); 2: train(100Hz),test(50Hz)

    args = parser.parse_args()
    return args

def iteration(data_loader,device,model,optim,task,train=True,mask=False):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_func=nn.CrossEntropyLoss()
    loss_list = []
    acc_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, _, action, people, timestamp in pbar:
        x = x.float().to(device)
        timestamp = timestamp.float().to(device)
        if task == "action":
            label = action.long().to(device)
        elif task == "fall":
            label = action.long().to(device)
            label[label>1] = 1
        elif task == "people":
            label = people.long().to(device)
        else:
            print("ERROR")
            exit(-1)

        input = copy.deepcopy(x)
        non_pad = (input != pad).float().to(device)
        avg = torch.sum(input * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((input - avg) ** 2) * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        input = (input - avg) / (std + 1e-8)
        non_pad=non_pad.bool()
        batch_size, seq_len, carrier_num = input.shape
        rand_word = torch.randn((batch_size, seq_len, carrier_num)).to(device)
        input[~non_pad]=rand_word[~non_pad]

        if mask :#and train:
            loss_mask = torch.zeros([batch_size, seq_len]).to(device)
            chosen_num_min = int(seq_len * 0.1)
            chosen_num_max = int(seq_len * 0.3)
            num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
            row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
            col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
            loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
            input[loss_mask.bool()] = rand_word[loss_mask.bool()]

        y = model(x,timestamp)
        loss = loss_func(y, label)
        output = torch.argmax(y, dim=-1)
        acc = torch.sum(output == label) / batch_size

        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()

        loss_list.append(loss.item())
        acc_list.append(acc.item())

    return np.mean(loss_list),np.mean(acc_list)

def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert=CSIBERT(bertconfig,args.carrier_dim).to(device)
    if not args.no_pretrain:
        csibert.load_state_dict(torch.load(args.path))
    if len(cuda_devices) > 1 and not args.cpu:
        csibert = nn.DataParallel(csibert, device_ids=cuda_devices)
    model=Sequence_Classifier(csibert,args.class_num).to(device)
    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.mode == 0:
        if args.recover_data:
            train_data, test_data = load_data(data_path=args.data_path,train_prop=args.train_prop,magnitude_path=args.magnitude_path)
        else:
            if args.random_input:
                train_data, test_data = load_data_random(data_path=args.data_path, train_prop=args.train_prop,
                                                         trainset_num=2000, testset_num=150, min_len=args.max_len,
                                                         max_len=args.max_len * 3, length=args.max_len)
            else:
                train_data, test_data = load_data_random(data_path=args.data_path, train_prop=args.train_prop,
                                                         trainset_num=2000, testset_num=150, min_len=args.max_len,
                                                         max_len=args.max_len, length=args.max_len)
    elif args.mode == 1:
        if args.dataset=="WiFall":
            train_data1, test_data1 = load_data_random(data_path=args.data_path, train_prop=args.train_prop, trainset_num=2000,
                                                          testset_num=150, min_len=args.max_len, max_len=args.max_len,
                                                          length=args.max_len, gap=1)
            train_data2, _ = load_data_random(data_path=args.data_path, train_prop=args.train_prop, trainset_num=2000,
                                                          testset_num=150, min_len=args.max_len, max_len=args.max_len,
                                                          length=args.max_len, gap=2)
            _, test_data2 = load_data_random(data_path=args.data_path, train_prop= 1 - args.train_prop, trainset_num=2000,
                                                          testset_num=150, min_len=args.max_len, max_len=args.max_len,
                                                          length=args.max_len, gap=2)
        else:
            train_data1, _, test_data1 = load_data_random(data_path=args.data_path,train_prop=args.train_prop/2,valid_prop=args.train_prop/2,trainset_num=2000,testset_num=150,min_len=args.max_len,max_len=args.max_len,length=args.max_len,gap=1)
            _, train_data2, test_data2 = load_data_random(data_path=args.data_path,train_prop=args.train_prop/2,valid_prop=args.train_prop/2,trainset_num=2000,testset_num=150,min_len=args.max_len,max_len=args.max_len,length=args.max_len,gap=2)
        train_data = ConcatDataset([train_data1, train_data2])
        test_data = ConcatDataset([test_data1, test_data2])
    elif args.mode == 2:
        if args.dataset=="WiFall":
            train_data, _ = load_data_random(data_path=args.data_path, train_prop=args.train_prop, trainset_num=2000,
                                             testset_num=150, min_len=args.max_len, max_len=args.max_len,
                                             length=args.max_len, gap=1)
            _, test_data = load_data_random(data_path=args.data_path, train_prop=1-args.train_prop, trainset_num=2000,
                                            testset_num=150, min_len=args.max_len, max_len=args.max_len,
                                            length=args.max_len, gap=2)
        else:
            train_data, _ = load_data_random(data_path=args.data_path,train_prop=args.train_prop,trainset_num=2000,testset_num=150,min_len=args.max_len,max_len=args.max_len,length=args.max_len,gap=1)
            _, test_data = load_data_random(data_path=args.data_path,train_prop=args.train_prop,trainset_num=2000,testset_num=150,min_len=args.max_len,max_len=args.max_len,length=args.max_len,gap=2)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    best_loss = 1e8
    best_acc = 0
    loss_epoch = 0
    acc_epoch = 0
    j = 0

    while True:
        j+=1
        if args.freeze:
            csibert.eval()
            for param in csibert.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        loss, acc = iteration(train_loader,device,model,optim,args.task,train=True,mask=args.mask)
        log = "Epoch {} | Train Loss {:06f} ,  Train Acc {:06f} | ".format(j, loss, acc)
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log)

        loss, acc = iteration(test_loader,device,model,optim,args.task,train=False,mask=args.mask)
        log = "Test Loss {:06f}, Test Acc {:06f} ".format(loss,acc)
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log+"\n")

        if acc>=best_acc or loss<=best_loss:
            torch.save(model.state_dict(), args.task+".pth")

        if acc>=best_acc:
            best_acc=acc
            acc_epoch=0
        else:
            acc_epoch+=1
        if loss<best_loss:
            best_loss=loss
            loss_epoch=0
        else:
            loss_epoch+=1
        if acc_epoch>=args.epoch and loss_epoch>=args.epoch:
            break
        print("Acc Epoch {:}, Loss Epcoh {:}".format(acc_epoch,loss_epoch))

if __name__ == '__main__':
    main()