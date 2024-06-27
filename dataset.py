from torch.utils.data import Dataset
import numpy as np
import copy
import pickle
from torch.utils.data import DataLoader
import torch


pad=-1000
time_gap=10000

class CSI_dataset(Dataset):
    def __init__(self, magnitudes, phases=None, timestamp=None, label_action=None, label_people=None):
        super().__init__()
        self.magnitudes = magnitudes
        self.phases = phases
        self.timestamp=timestamp
        self.label_action = label_action
        self.label_people = label_people
        self.num=self.magnitudes.shape[0]
        if self.phases is None:
            self.phases = [-1] * self.num
        if self.timestamp is None:
            self.timestamp = [-1] * self.num
        if self.label_action is None:
            self.label_action = [-1] * self.num
        if self.label_people is None:
            self.label_people = [-1] * self.num

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.magnitudes[index],self.phases[index], self.label_action[index], self.label_people[index], self.timestamp[index]


def load_data(data_path="./data",train_prop=None,valid_prop=None, data_num=None,magnitude_path=None):
    if magnitude_path is not None:
        # magnitude = np.load(data_path+"/"+magnitude_path).astype(np.float32)
        # magnitude = np.load(magnitude_path+"/magnitude.npy").astype(np.float32)
        magnitude = np.load(magnitude_path).astype(np.float32)
    else:
        magnitude = np.load(data_path+"/magnitude.npy").astype(np.float32)
    people=np.load(data_path+"/people.npy").astype(np.int64)
    action=np.load(data_path+"/action.npy").astype(np.int64)
    phase=np.load(data_path+"/phase.npy").astype(np.float32)
    timestamp=np.load(data_path+"/timestamp.npy").astype(np.float32)
    if train_prop is None:
        if data_num is None:
            return CSI_dataset(magnitude, phase, timestamp, action, people)
        else:
            return CSI_dataset(magnitude[:data_num], phase[:data_num], timestamp[:data_num], action[:data_num], people[:data_num])
    else:
        a = np.zeros_like(people)
        num=[]
        current_num=0
        current_action=None
        for i in range(action.shape[0]):
            if action[i]==current_action:
                current_num+=1
            else:
                current_action = action[i]
                if current_action is None:
                    current_num+=1
                else:
                    num.append(current_num)
                    current_num=0
        num.append(current_num)
        if valid_prop is None:
            current_num=0
            for i in range(len(num)):
                a[current_num:current_num+int(num[i]*train_prop)]=1
                current_num+=num[i]
            b=1-a
            a = a.astype(bool)
            b = b.astype(bool)
            return CSI_dataset(magnitude[a], phase[a], timestamp[a], action[a], people[a]), CSI_dataset(magnitude[b], phase[b], timestamp[b], action[b], people[b])
        else:
            current_num=0
            b = np.zeros_like(people)
            for i in range(len(num)):
                a[current_num:current_num+int(num[i]*train_prop)]=1
                b[current_num+int(num[i]*train_prop):current_num+int(num[i]*(train_prop+valid_prop))]=1
                current_num+=num[i]
            c=1-a-b
            a = a.astype(bool)
            b = b.astype(bool)
            c = c.astype(bool)
            return CSI_dataset(magnitude[a], phase[a], timestamp[a], action[a], people[a]), CSI_dataset(magnitude[b], phase[b], timestamp[b], action[b], people[b]), CSI_dataset(magnitude[c], phase[c], timestamp[c], action[c], people[c])


class CSI_dataset_random(Dataset):
    def __init__(self, magnitudes, phases=None, timestamp=None, label_action=None, label_people=None, num=2000, min_len=100, max_len=300, length=100):
        super().__init__()
        self.magnitudes = magnitudes
        self.phases = phases
        self.timestamp=timestamp
        self.label_action = label_action
        self.label_people = label_people
        self.num=num
        self.min_len=min_len
        self.max_len=max_len
        self.length=length
        if self.phases is None:
            self.phases = copy.deepcopy(magnitudes)
        if self.timestamp is None:
            self.timestamp = copy.deepcopy(magnitudes)
        if self.label_action is None:
            self.label_action = [-1] * len(magnitudes)
        if self.label_people is None:
            self.label_people = [-1] * len(magnitudes)


    def __len__(self):
        return self.num

    def __getitem__(self, index):
        i=np.random.randint(0,len(self.magnitudes))
        magnitude=self.magnitudes[i]
        phase=self.phases[i]
        timestamp=self.timestamp[i]
        action=self.label_action[i]
        people=self.label_people[i]

        while magnitude.shape[0]<=self.max_len:
            i = np.random.randint(0, len(self.magnitudes))
            magnitude = self.magnitudes[i]
            phase = self.phases[i]
            timestamp = self.timestamp[i]
            action = self.label_action[i]
            people = self.label_people[i]

        l=np.random.randint(0,magnitude.shape[0]-self.max_len)
        r=l+np.random.randint(self.min_len,self.max_len+1)
        magnitude_full = magnitude[l:r]
        phase_full = phase[l:r]
        timestamp_full = timestamp[l:r]

        sampled_indices = np.random.choice(len(magnitude_full), size=self.length, replace=False)
        sampled_indices = np.sort(sampled_indices)
        # print(sampled_indices)
        magnitude = magnitude_full[sampled_indices]
        phase = phase_full[sampled_indices]
        timestamp = timestamp_full[sampled_indices]

        return magnitude,phase,action,people,timestamp



def load_data_random(data_path="./data/data_sequence.pkl",train_prop=None,valid_prop=None,trainset_num=2000,validset_num=150,testset_num=150,min_len=100,max_len=300,length=100,gap=1):
    with open(data_path, 'rb') as f:
        csi = pickle.load(f)

    action_list = []
    people_list = []
    timestamp = []
    magnitudes = []
    phases = []

    for data in csi:
        local_time = data['time']
        magnitude = data['magnitude']
        phase = data['phase']
        people = data['people']
        action = data['action']
        if gap!=1:
            local_time=local_time[::gap]
            magnitude=magnitude[::gap,:]
            phase=phase[::gap,:]
        action_list.append(action)
        people_list.append(people)
        magnitudes.append(magnitude)
        timestamp.append(local_time)
        phases.append(phase)
    if train_prop is None:
        return CSI_dataset_random(magnitudes, phases, timestamp, action_list, people_list, num=trainset_num, min_len=min_len, max_len=max_len,length=length)
    elif valid_prop is None:
        train_timestamp = []
        train_magnitudes = []
        train_phases = []
        test_timestamp = []
        test_magnitudes = []
        test_phases = []
        for i in range(len(action_list)):
            num=magnitudes[i].shape[0]
            train_num=int(num*train_prop)
            train_timestamp.append(timestamp[i][:train_num])
            train_magnitudes.append(magnitudes[i][:train_num])
            train_phases.append(phases[i][:train_num])
            test_timestamp.append(timestamp[i][train_num:])
            test_magnitudes.append(magnitudes[i][train_num:])
            test_phases.append(phases[i][train_num:])
        return CSI_dataset_random(train_magnitudes, train_phases, train_timestamp, action_list, people_list, num=trainset_num, min_len=min_len, max_len=max_len,length=length),CSI_dataset_random(test_magnitudes, test_phases, test_timestamp, action_list, people_list, num=testset_num, min_len=min_len, max_len=max_len,length=length)
    else:
        train_timestamp = []
        train_magnitudes = []
        train_phases = []
        valid_timestamp = []
        valid_magnitudes = []
        valid_phases = []
        test_timestamp = []
        test_magnitudes = []
        test_phases = []
        for i in range(len(action_list)):
            num=magnitudes[i].shape[0]
            train_num=int(num*train_prop)
            valid_num=int(num*valid_prop)+train_num
            train_timestamp.append(timestamp[i][:train_num])
            train_magnitudes.append(magnitudes[i][:train_num])
            train_phases.append(phases[i][:train_num])
            valid_timestamp.append(timestamp[i][train_num:valid_num])
            valid_magnitudes.append(magnitudes[i][train_num:valid_num])
            valid_phases.append(phases[i][train_num:valid_num])
            test_timestamp.append(timestamp[i][valid_num:])
            test_magnitudes.append(magnitudes[i][valid_num:])
            test_phases.append(phases[i][valid_num:])
        return CSI_dataset_random(train_magnitudes, train_phases, train_timestamp, action_list, people_list, num=trainset_num, min_len=min_len, max_len=max_len,length=length), CSI_dataset_random(valid_magnitudes, valid_phases, valid_timestamp, action_list, people_list, num=validset_num, min_len=min_len, max_len=max_len,length=length),CSI_dataset_random(test_magnitudes, test_phases, test_timestamp, action_list, people_list, num=testset_num, min_len=min_len, max_len=max_len,length=length)

if __name__ == '__main__':
    train_data, test_data = load_data_random(data_path="./data/data_sequence.pkl",train_prop=0.9,trainset_num=2000,testset_num=150,min_len=100,max_len=100,length=100)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    data_iter = iter(train_loader)
    magnitude, phase, action, people, timestamp = next(data_iter)
    # print(magnitude[0])
    # print(action)
    # print(people)
    # print(timestamp[0])
    timestamp_sort,_=torch.sort(timestamp,dim=-1)
    print((timestamp_sort==timestamp).all())
