<span id="top"></span>
- [File Struct](#file-struct)
- [DataSet construct](#dataset-construct)
  - [pytorch dataset](#pytorch-dataset)
- [Net construct](#net-construct)
  - [network building](#network-building)
  - [loss function](#loss-function)
- [train and test](#train-and-test)
  - [train](#train)
  - [test](#test)
- [Utils](#utils)
  - [logging](#logging)
- [Porblem collection](#porblem-collection)

# File Struct

```text
|-- project
    |-- dataset
    |   |-- preprocess.py
    |   |__ pytorch_dataset.py
    |-- net model
    |   |-- net_architecture.py
    |   |__ loss_function.py
    |-- train and test
    |   |-- train.py
    |   |__ test.py
    |__ utils
        |__ utils.py
```

# DataSet construct

## pytorch dataset

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

## network building
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

## loss function

```python
import os
import torch
import torch.nn as nn


"""方法一：直接利用torch.Tensor提供的接口"""
import torch.nn.functional as F


class TensorLoss(nn.Module):
    def __init__(self, ):
        super(TensorLoss, self).__init__()

    def forward(self, prediction, target):
        loss = F.cross_entropy(prediction, target)

        return loss


"""方法二：利用PyTorch的numpy/scipy扩展"""
# 等待完善细节
from torch.autograd import Function

class GradLoss(Function):
    def forward(input_tensor):
        tensor = input_tensor.numpy()
        ......  # 其它 numpy/scipy 操作
        result = ......
        return torch.Tensor(result)

    def backward(grad_output):
        pass
"""方法三:写一个PyTorch的C扩展"""
# 等待完善

```

[Back to top](#top)

# train and test

## train
```python
import os
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_s
from torch.utils.data import Dataloader

from dataset import Dataset
from network import NetModel


manualSeed = 1
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)


def init_parser():
    parsers = argparse.ArgumentParser(description="This is parser example.")
    parsers.add_argument('-d', '--dataset',
                         default="MSRA",
                         choices=["MSRA", "VOC"],
                         type=str,
                         help="load which dataset [default: MSRA]. [choices:['MSRA', 'VOC']]")
    parsers.add_argument('-b', '--Batch_Size',
                         default=8,
                         type=str,
                         help="input batch size [default: 8]")
    parsers.add_argument('-e', '--epochs',
                         default=30,
                         type=int,
                         help="training epoch [default: 30]")
    parsers.add_argument('-start',
                         default=0,
                         type=int,
                         help="the start epoch of the training [default: 0]")
    parsers.add_argument('-c', '--checkpoint',
                         default='./checkpoint',
                         type=str, help="the path to save checkpoint [default: ./checkpoint]")
    parsers.add_argument('-m', '--model',
                         default=None,
                         type=str,
                         help="pretrained model path, if is none,train a new model [default: None]")
    parsers.add_argument('w', '--workers',
                         type=int,
                         default=0,
                         help='number of data loading workers [default: None')
    parsers.add_argument('-lr',
                         default=1e-2,
                         type=float,
                         help="Initial learning rate [default: 1e-2]")
    args = parsers.parse_args()

    return args


def train(args):
    # create checkpoint save path
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    # loading data
    print("Loading data")
    train_data = Dataset(root=args.root, mode='train', args)
    train_dataloder = Dataloder(train_data,
                                batch_size=args.Batch_Size,
                                shuffle=True, num_works=args.works)

    model = NetModel(args)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    model.cuda()

    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.5, 0.999), eps=1e-06)
    # change the learning rate at epoch 20
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    model.train()
    for epoch in range(args.start, args.epoch):
        scheduler.step(epoch)
        total_loss = 0
        for idx, data in enumerate(tqdm(train_dataloder, 0)):
            img, label = data['data'], data['label']
            img, label = img.cuda(), label.cuda()

            predict = model(img)

            optimizer.zero_grad()
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()

            # thus loss is a cuda(gpu) variable(with grad),
            # change it to a cpu no grad tensor first to for unrelated operation
            # loss.detach for pytorch version> 1.0
            # for lower version, use loss.item() or loss.data[0]
            total_loss += loss.detach().cpu()

        print(f"Average loss in epoch {epoch:02d} is {total_loss/idx:.4f}.")

        # f-format only available when python version > 3.6
        # when use lower version, use ["epoch_%02d.pth"%epoch] instead
        torch.save(model.state_dict(), os.path.join(args.checkpoint, f"epoch_{epoch:02d}.pth"))

```
1. pytorch 常用的loss 函数
   很多的 loss 函数都有 size_average 和 reduce 两个布尔类型的参数，因为一般损失函数都是直接计算 batch 的数据，因此返回的 loss 结果都是维度为 (batch_size, ) 的向量。
   - 如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss；
   - 如果 reduce = True，那么 loss 返回的是标量 
     - 如果 size_average = True，返回 loss.mean();
     - 如果 size_average = True，返回 loss.sum();
   1. **L1** 
    $$\text{loss}(\mathbf{x}_i, \mathbf{y}_i) = |\mathbf{x}_i - \mathbf{y}_i|$$
    要求 x 和 y 的维度要一样（可以是向量或者矩阵），得到的 loss 维度也是对应一样的。这里用下标 i 表示第 i 个元素。
   2. **MSE(均方损失函数)**
    $$\text{loss}(\mathbf{x}_i, \mathbf{y}_i) = (\mathbf{x}_i - \mathbf{y}_i)^2$$

   3. **BCE** 
    $$\text{loss}(\mathbf{x}_i, \mathbf{y}_i) = - \boldsymbol{w}_i \left[\mathbf{y}_i \log \mathbf{x}_i + (1-\mathbf{y}_i)\log(1-\mathbf{x}_i) \right ]$$
    二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数。
   4. **CrossEntropy**
    $$\begin{aligned}
    \text{loss}(\mathbf{x}, \text{label}) &= - \boldsymbol{w}_{\text{label}}\log \frac{e^{\mathbf{x}_{\text{label}}}}{\sum_{j=1}^N e^{\mathbf{x}_j}} \\
    &= \boldsymbol{w}_{\text{label}} \left[ -\mathbf{x}_{\text{label}} + \log \sum_{j=1}^N e^{\mathbf{x}_j} \right]
    \end{aligned}$$
    多分类用的交叉熵损失函数，用这个 loss 前面不需要加 Softmax 层,这里限制了 target 类型为 torch.LongTensr，而且不是多标签意味着标签是 one-hot 编码的形式，即只有一个位置是 1，其他位置都是 0.
   5. **NLL**
    $$\text{loss}(\mathbf{x}, \text{label}) = - \mathbf{x}_{\text{label}}$$
    用于多分类的负对数似然损失函数（Negative Log Likelihood）在前面接上一个 nn.LogSoftMax 层就等价于交叉熵损失了.
2. **SSIM**
   结构相似性指数（structural similarity index，SSIM）, 出自参考文献[1]，用于度量两幅图像间的结构相似性。和被广泛采用的L2 loss不同，SSIM和人类的视觉系统（HVS）类似，对局部结构变化的感知敏感。
   SSIM分为三个部分：照明度、对比度、结构，分别如下公式所示：
   $$\begin{align*}
       l(x,y)&=\frac{2\mu_x\mu_y+c_1}{\mu_x^2+\mu_y^2+c_1}\\
       c(x,y)&=\frac{2\sigma_x\sigma_y+c_2}{\sigma_x^2+\sigma_x^2+c_2}\\
       s(x,y)&=\frac{\sigma_{xy}+c_3}{\sigma_x+\sigma_y+c_3}
   \end{align*}$$
   将三个式子汇总到一起就是SSIM：
   $$SSIM(x,y)=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}$$
   其中，上式各符号分别为图像x和y的均值、方差和它们的协方差，显而易见，不赘述。$c_{1}=(k_{1}L)^{2}$，$c_{2}=(k_{2}L)^{2}$为常数。一般默认$k_{1}=0.01$，$k_{2}=0.03$. L为像素值的动态范围，如8-bit深度的图像的L值为2^8-1=255.
    SSIM值越大代表图像越相似，当两幅图像完全相同时，$SSIM=1$。所以作为损失函数时，应该要取负号，例如采用 $loss = 1 - SSIM$ 的形式。
    代码见[SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py)

[Back to top](#top)

## test
```python
import os
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_s
from torch.utils.data import Dataloader

from dataset import Dataset
from network import NetModel


manualSeed = 1
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)


def init_parser():
    parsers = argparse.ArgumentParser(description="This is parser example.")
    parsers.add_argument('-d', '--dataset',
                         default="MSRA",
                         choices=["MSRA", "VOC"],
                         type=str,
                         help="load which dataset [default: MSRA]. [choices:['MSRA', 'VOC']]")
    parsers.add_argument('-b', '--Batch_Size',
                         default=8,
                         type=str,
                         help="input batch size [default: 8]")
    parsers.add_argument('-e', '--epochs',
                         default=30,
                         type=int,
                         help="training epoch [default: 30]")
    parsers.add_argument('-start',
                         default=0,
                         type=int,
                         help="the start epoch of the training [default: 0]")
    parsers.add_argument('-c', '--checkpoint',
                         default='./checkpoint',
                         type=str, help="the path to save checkpoint [default: ./checkpoint]")
    parsers.add_argument('-m', '--model',
                         default=None,
                         type=str,
                         help="pretrained model path, if is none,train a new model [default: None]")
    parsers.add_argument('w', '--workers',
                         type=int,
                         default=0,
                         help='number of data loading workers [default: None')
    parsers.add_argument('-lr',
                         default=1e-2,
                         type=float,
                         help="Initial learning rate [default: 1e-2]")
    args = parsers.parse_args()

    return args


def test(args):

    # loading data
    print("Loading data")
    test_data = Dataset(root=args.root, mode='test', args)
    test_dataloder = Dataloder(test_data,
                               batch_size=args.Batch_Size,
                               shuffle=True, num_works=args.works)

    model = NetModel(args)
    assert args.model is not None, print("please select trained model")
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.test()
    for epoch in range(args.start, args.epoch):
        scheduler.step(epoch)
        total_loss = 0
        for idx, data in enumerate(tqdm(test_dataloder, 0)):
            img, label = data['data'], data['label']
            img, label = img.cuda(), label.cuda()

            predict = model(img)

            loss = criterion(predict, label)

            # thus loss is a cuda(gpu) variable(with grad),
            # change it to a cpu no grad tensor first to for unrelated operation
            # loss.detach for pytorch version> 1.0
            # for lower version, use loss.item() or loss.data[0]
            total_loss += loss.detach().cpu()

            # if need prediction data,
            result = predict.detach().cpu().numpy()

        print(f"Average loss in epoch {epoch:02d} is {total_loss/idx:.4f}.")

```

[Back to top](#top)

# Utils

Additional part, recommend subject: logging 

## logging

[logging](../Python/logging.md)

[Back to top](#top)

# Porblem collection

[problems](./pytorch_problems.md)

[Back to top](#top)
