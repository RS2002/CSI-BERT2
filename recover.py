from model import CSIBERT,Token_Classifier
from transformers import BertConfig
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import load_data
import torch.nn as nn
import copy
import numpy as np
import time

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
    parser.add_argument('--magnitude_path', type=str, default="./data/WiGesture/intern/0")
    parser.add_argument('--data_path', type=str, default="./data/WiGesture")
    parser.add_argument('--parameter', type=str, default="./pretrain.pth")

    args = parser.parse_args()
    return args

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
    if len(cuda_devices) > 1 and not args.cpu:
        csibert = nn.DataParallel(csibert, device_ids=cuda_devices)

    model = Token_Classifier(csibert, args.carrier_dim).to(device)
    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)
    if args.parameter is not None:
        model.load_state_dict(torch.load(args.parameter))

    model.eval()
    torch.set_grad_enabled(False)

    dataset = load_data(data_path=args.data_path,magnitude_path=args.magnitude_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pbar = tqdm.tqdm(data_loader, disable=False)
    output1 = None
    output2 = None
    start_time = time.time()
    model.eval()
    for x, _, _, _, timestamp in pbar:
        x = x.float().to(device)
        timestamp = timestamp.float().to(device)
        input = copy.deepcopy(x)

        # standard
        non_pad = (input != pad).float().to(device)
        avg = torch.sum(input * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((input - avg) ** 2) * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        input = (input - avg) / (std + 1e-8)

        non_pad=non_pad.bool()
        batch_size, seq_len, carrier_num = input.shape
        rand_word = torch.randn((batch_size, seq_len, carrier_num)).to(device)
        input[~non_pad]=rand_word[~non_pad]

        y = model(input, timestamp)
        y = y * std + avg
        non_pad = non_pad.float()
        y2 = x * non_pad + y * (1-non_pad)

        if output1 is None:
            output1=y
            output2=y2
        else:
            output1=torch.cat([output1,y],dim=0)
            output2=torch.cat([output2,y2],dim=0)

    replace = output1.cpu().numpy()
    recover = output2.cpu().numpy()
    np.save(args.magnitude_path+"/replace.npy", replace)
    np.save(args.magnitude_path+"/recover.npy", recover)
    end_time = time.time()
    print(f"Time Cost: {end_time - start_time} s")

if __name__ == '__main__':
    main()


