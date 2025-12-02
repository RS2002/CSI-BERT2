# CSI-BERT2

**Article:** Zijian Zhao, Fanyi Meng, Zhonghao Lyu, Hang Li, XiaoYang Li, Guangxu Zhu*, "[CSI-BERT2: A BERT-inspired Framework for Efficient CSI Prediction and Classification in Wireless Communication and Sensing](https://arxiv.org/abs/2412.06861)", IEEE Transactions on Mobile Computing (TMC), 2025

Upgraded version of [Official Repository for The Paper, Finding the Missing Data: A BERT-inspired Approach Against Package Loss in Wireless Sensing](https://github.com/RS2002/CSI-BERT).

**Notice:** We have uploaded our model, pre-trained parameters ([RS2002/WiGesture · Datasets at Hugging Face](https://huggingface.co/datasets/RS2002/WiGesture)), and dataset ([RS2002/WiCount · Datasets at Hugging Face](https://huggingface.co/datasets/RS2002/WiCount)) to Hugging Face.

![](./img/main.png)



## 1. Data

### 1.1 Dataset

Public Dataset: [WiGesture](http://www.sdp8.net/Dataset?id=5d4ee7ca-d0b0-45e3-9510-abb6e9cdebf9), [WiFall](https://github.com/RS2002/KNN-MMD/tree/main/WiFall)

Proposed Dataset: WiCount (./WiCount)



### 1.2 Data Preparation

Refer to [RS2002/CSI-BERT: Official Repository for The Paper, Finding the Missing Data: A BERT-inspired Approach Against Package Loss in Wireless Sensing (github.com)](https://github.com/RS2002/CSI-BERT)



## 2. Train

### 2.1 Pre-train

```shell
python pretrain.py --GAN --data_path <data path>
```

If you do not want to use the discriminator, you can delete the `--GAN`, it keeps the same in the following.



### 2.2 Fine-tune

#### 2.2.1 CSI Prediction Task

```shell
python prediction.py --GAN --data_path <data path> --parameters <fold path of the whole pre-trained models>
```



#### 2.2.2 CSI Sensing Task

```shell
python finetune.py --data_path <data path> --class_num <class num> --task <task name> --path <parameter path of the backbone> --mode <mode>
```

The mode can be set as 0, 1, or 2, corresponding to three experiments in our paper:
0: Training Set (100Hz), Testing Set (100Hz)
1: Training Set (100Hz+50Hz), Testing Set (100Hz+50Hz)
2: Training Set (100Hz), Testing Set (50Hz)

You can also change the `gap` parameter in `load_data_random` function to get more sampling rate.



The task name can be set as "action", "fall", or "people", representing different tasks when using different datasets:
WiGesture: action (gesture recognition), people (people identification)
WiFall: action (action recognition), fall (fall detection), people (people identification)
WiCount: people (people number estimation)



### 2.3 Infer

#### 2.3.1 CSI Recovery Task

```shell
python recover.py  --data_path <data path> --parameters <parameter path of the pre-trained recoverer>
```



#### 2.3.2 CSI Prediction Task

```shell
python prediction.py  --data_path <data path> --parameters <fold path of the whole pretrained models> --eval_percent <the percentage of CSI sequence to be predicted>
```

 

## 3. Notice

The current version of our code does not support multiple GPUs. Please specify only one GPU or fix the relevant code. We would appreciate if you could share the code that can solve this problem.



## 4. Reference

```
@article{zhao2024mining,
  title={CSI-BERT2: A BERT-inspired Framework for Efficient CSI Prediction and Classification in Wireless Communication and Sensing},
  author={Zhao, Zijian and Meng, Fanyi and Lyu, Zhonghao and Li, Hang and Li, Xiaoyang and Zhu, Guangxu},
  journal={arXiv preprint arXiv:2412.06861},
  year={2024}
}
```

