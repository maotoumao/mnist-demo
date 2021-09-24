import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils



# 定义网络
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, 5),  # 输入是1个神经元，输出10个，卷积核大小为5
            nn.ReLU(),  # relu
            nn.MaxPool2d((2, 2)),  # 2x2 池化
            nn.Conv2d(10, 20, 3),  # 图像大小变成了10x10，输出有20个通道（也就是20个神经元）
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 10),  # 10分类
            nn.LogSoftmax(1)  # log(softmax)
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out


# 定义训练函数
def train(model, device, train_data_loader, optimizer):
    # 训练模式
    model.train()
    for index, (data, label) in enumerate(train_data_loader):
        data, label = data.to(device), label.to(device)
        # 清空梯度
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, label)
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        if index % 50 == 0:
            print('Batch Index: {}  Loss: {}'.format(index, loss.item()))


# 输出模型
def transformToOnnx(model, batch_size, name='mnist.onnx'):
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)
    torch.onnx.export(model, x, name, export_params=True, opset_version=11, do_constant_folding=True, input_names=[
                      'input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


# 常量
BATCH_SIZE = 512
EPOCHS = 10
DEVICE = 'cpu'

# 加载训练集，并标准化
train_data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 实例化模型
model = ConvNet().to(DEVICE)

# 定义优化函数
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(EPOCHS):
    print('------ EPOCH: {} ------'.format(epoch + 1))
    train(model, DEVICE, train_data_loader, optimizer)

# 保存模型
transformToOnnx(model, BATCH_SIZE)
