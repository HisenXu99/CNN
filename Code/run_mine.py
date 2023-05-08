from lenet_mine import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import onnx
from data import ObjectLandmarksDataset

viz = visdom.Visdom()


# label_file="/home/hisen/Project/Data/CR5_picture/test.txt"
# root_dir  ="/home/hisen/Project/Data/CR5_picture/"
label_file="/remote-home/2230728/CNN/CR5_picture/test.txt"
root_dir  ="/remote-home/2230728/CNN/CR5_picture/"
data_train = ObjectLandmarksDataset(label_file,root_dir)
data_train_loader = DataLoader(data_train, batch_size=10, shuffle=False, num_workers=8)

net = LeNet5()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()
    return loss_list[-1]


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    loss=train(epoch)
    #test()

    dummy_input = torch.randn(10, 3, 360, 500, requires_grad=True).cuda()
    # dummy_input=dummy_input.to(device)
    torch.onnx.export(net, dummy_input, "lenet.onnx")

    onnx_model = onnx.load("lenet.onnx")
    onnx.checker.check_model(onnx_model)

    #看训练效果
    with open("/remote-home/2230728/CNN/record.txt","a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(epoch))
        file.write("   ")
        file.write(str(loss)+"\n")


def main():
    for e in range(1, 10000):
        train_and_test(e)


if __name__ == '__main__':
    main()
