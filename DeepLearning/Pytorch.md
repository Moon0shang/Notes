<span id="top"></span>
- [File Struct](#file-struct)
- [DataSet construct](#dataset-construct)
- [Net construct](#net-construct)
- [train and test](#train-and-test)
  - [train](#train)
  - [test](#test)
- [Utils](#utils)
  - [logging](#logging)

# File Struct

```text
|-- project
    |-- dataset
    |   |-- preprocess.py
    |   |__ pytorch_dataset.py
    |-- net model
    |   |__ net_architecture.py
    |-- train and test
    |   |-- train.py
    |   |__ test.py
    |__ utils
        |__ utils.py
```

# DataSet construct

pytorch dataset:
```python
import torch
import torch.utils.data as data
from torchvision import transforms

import numpy as np


class Dataset(data.Dataset):
    def __init__(self, root: str, mode: str):
        super(Dataset, self).__init__()
        self.root = root
        self.mode = mode
        self.transform = transforms.Compose(
            [Normolization("modify args"),
             ToTensor()])

    def __len__(self):
        return len("dataset size")

    def __getitem__(self, index: int):
        sample = {'data': data[index], 'label': label[index]}
        return self.transform(sample)
    
    def _other_function(self,):
        """other function for reading the data or a data processing"""
        pass


class Normolization():
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0, 0, 0), std=(0, 0, 0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict()):
        data = sample['data']
        label = sample['label']
        data = np.array(data).astype(np.float32)
        label = np.array(label).astype(np.float32)
        data -= self.mean
        data /= self.std

        return {'data': data, 'label': label}


class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: dict()):
        
        data = sample['data']
        label = sample['label']
        # swap color axis 
        # numpy image: H x W x C
        # torch image: C X H X W
        # data = np.array(data).astype(np.float32).transpose((2, 0, 1))
        # label = np.array(label).astype(np.float32)
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()

        return {'data': data, 'label': label}
    
```

[Back to top](#top)

# Net construct

```python
import torch
import torch.nn as nn

from collections import OrderedDict

'''方法一：直接构建网络，适用于小型网络'''


class Minist(nn.Module):
    def __init__(self):
        super(Minist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxp = nn.MaxPool2d(stride=2, kernel_size=2)
        self.liner1 = nn.Linear(14 * 14 * 128, 1024)
        self.relu3 = nn.ReLU()
        self.dp = nn.Dropout(p=0.5)
        self.liner2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxp(self.relu2(self.conv2(x)))
        x = x.view(-1, 14 * 14 * 128)
        x = self.dp(self.relu3(self.liner1(x)))
        x = self.liner2(x)


'''方法二： 使用 nn.Sequential'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class NetSeq(nn.Module):
    def __init__(self,):
        super(NetSeq, self).__init__()
        self.vgg = self._make_layers('VGG16')
        self.net1 = nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.BatchNorm2d(3),
            nn.ReLU())
        self.net2 = nn.Sequential()
        self.net2.add_module('conv', nn.Conv2d(3, 3, 3))
        self.net2.add_module('batchnorm', nn.BatchNorm2d(3))
        self.net2.add_module('activation_layer', nn.ReLU())
        self.net3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 3, 3)),
            ('batchnorm', nn.BatchNorm2d(3)),
            ('activation_layer', nn.ReLU())]))
        # net3.conv = net3[0]

    def forward(self, x):
        x = self.vgg(x)
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        return x

    def _make_layers(self, cfg: dict(list())):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3,
                                     padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        # 使用 * 对list进行解构
        return nn.Sequential(*layers)


'''方法三：使用nn.ModuleList'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class NetList(nn.Module):
    def __init__(self,):
        super(NetList, self).__init__()
        self.vgg = nn.ModuleList(self._make_layers(cfg["VGG16"]))
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.linears.extend([nn.Linear(layers_size, layers_size)
                             for i in range(1, self.num_layers-1)])
        self.linears.append(nn.Linear(layers_size, output_size))

    def forward(self, x):
        for layer in self.vgg:
            x = layer(x)
        for layer in self.linears:
            x = layer(x)

        return x

    def _make_layers(self, cfg: dict(list())):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3,
                                     padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return layers


''' main 函数'''
if __name__ == "__main__":
    net1 = Minist()
    print("direct construct.\n", net1)
    net2 = NetSeq()
    print("use Sequential.\n", net2)
    net3 = NetList()
    print("use ModuleList.\n", net3)

```
[Back to top](#top)

1. nn.Sequential和nn.ModuleList的区别:

    nn.Sequential类似于Keras中的贯序模型，它是Module的子类，在构建数个网络层之后会自动调用forward()方法，从而有网络模型生成。而nn.ModuleList仅仅类似于pytho中的list类型，只是将一系列层装入列表，并没有实现forward()方法，因此也不会有网络模型产生的副作用。

2. nn.ModuleList接受的必须是subModule类型，二次嵌套的list内部也必须额外使用一个nn.ModuleList修饰实例化，否则会无法识别类型而报错！

    ```python
    nn.ModuleList([
        nn.ModuleList([
            Conv(inp_dim + j * increase, oup_dim, 
            1, relu=False, bn=False) for j in range(5)]) 
            for i in range(nstack)])
    ```
3. 只初始化网络部分权重
   ```python
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
   ```
4. 只加载部分网络预训练权重
   ```python
    def _load_pretrained_model(self, pretrained: str):
        """pretrained: the path of the pretrained model, *.pkl/pth"""
        # load model from internet
        # import torch.utils.model_zoo as model_zoo
        # pretrained_dict = model_zoo.load_url('http://*.pth')
        pretrained_dict = torch.load(pretrained)
        state_dict = self.state_dict()

        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        # one more example @https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
   ```

[Back to top](#top)

# train and test

## train

[Back to top](#top)

## test


[Back to top](#top)

# Utils

Additional part, recommend subject: logging 

## logging

[logging](../Python/logging.md)

[Back to top](#top)