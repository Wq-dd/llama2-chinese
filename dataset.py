
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split



class PretrainDataset(Dataset):
    def __init__(self,data_path_lst,max_length=256,memmap=False):
        super().__init__()
        #
        if memmap:
            with open(data_path_lst[0],'r') as f:
                nbytes = f.seek(0,2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_lst[0],dtype=np.dtype('uint16'),shape=(flen//max_length,max_length))
        else:
            data_lst=[]
            for data_path in data_path_lst:
                with open(data_path,'rb') as f:
                    data=np.fromfile(f,dtype=np.uint16)
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length*int(len(data)/max_length)]
            #np.random.shuffle(data)
            self.data = data.reshape(-1,max_length)
        #
        print("memmap:{} train data.shape:{}".format(memmap,self.data.shape))
        print("downloading finished.....")

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.data[index]
        X=np.array(sample[:-1]).astype(np.int64)
        Y=np.array(sample[1:]).astype(np.int64)

        return torch.from_numpy(X),torch.from_numpy(Y)
#

if __name__=="__main__":
    data_path_list=[
        "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_1.bin",
        "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_2.bin",
        "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_3.bin",
        "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_4.bin",
        "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_5.bin",
        # './data/pretrain_data.bin'
        #'./data/baidubaike_563w.bin',
        #'./data/medical_book.bin',
        # './data/medical_encyclopedia.bin',
        # './data/medical_qa.bin',
        # './data/wiki.bin'
    ]
    train_ds = PretrainDataset(data_path_list, max_length=512, memmap=False)
    for x, y in train_ds:
        print(x)
    pass