import torch
from torch import nn
from torchvision import transforms, datasets
from torchinfo import summary
import torch.optim as optim
from tqdm import tqdm
from tearch_model import TeacherModel


# 设置随机种子，便于复现
torch.manual_seed(0)

device = torch.device("cudn" if torch.cuda.is_available() else "cpu")
print(device)

# 使用cudnn加速卷积运算
# torch.backends.cudnn.benchmark = True

# 载入训练集
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# 载入测试集
test_dataset = datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
# 生成dataloader
batch_size = 2
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


model = TeacherModel()
model = model.to(device)
summary(model)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 6
save_path = './TearchModel.pth'  # 保存权重的路径
for epoch in range(epochs):
    model.train()

    # 训练集上训练模型权重
    for data, labels in tqdm(train_loader):
        # 将数据和标签读入设备
        data = data.to(device)
        labels = labels.to(device)

        # 前向传播
        preds = model(data)
        loss = loss_function(preds, labels)

        # 反向传播，优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试集上评估模型性能
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            preds = model(test_data)
            predict = preds.max(1).indices
            num_correct += (predict == test_labels).sum()
            num_samples += predict.size(0)
        acc = (num_correct/num_samples).item()
    model.train()
    print("Epoch:{}\t Accuracy:{:.4f}".format(epoch+1, acc))


    torch.save(model.state_dict(), save_path)

print('Finished Training')

