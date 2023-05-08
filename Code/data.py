from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as transforms 
import torch

class ObjectLandmarksDataset(Dataset):
    """Object Landmarks dataset."""
    def __init__(self, txt_file, root_dir, transform=True):
        """
        Args:
            csv_file (string): 到达标注文件cvs的路径.
            root_dir (string): 所有图片的根目录.
            transform (callable, optional): （可选参数）对每一个样本进行转换.
        """
        self.name_list,self.labels_list=self.get_labels(txt_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_list)
    
    def get_labels(self,dir_path):
        labels_list = []  # 标签列表
        name_list=[]      # 名字列表
        with open(dir_path,'r',encoding = 'utf-8') as f:
            line = f.readline().strip('\n') # 读取第一行
            while line:
                line.rstrip()
                if(not line): continue   #空不记入
                information = line.split()
                label=[float(information[0]),float(information[1])]
                labels_list.append(label)
                line = f.readline().strip('\n') # 读取下一行
        for i in range(len(labels_list)):
            name_list.append(str(i+1)+".jpg")
        return name_list,labels_list

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.name_list[idx]) #第idx条数据的第一个字段，即文件名称
        image = cv2.imread(img_name)                           #读取图像数据
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        pose = self.labels_list[idx]                     #获取图片的标签
        pose= torch.Tensor(pose)
        if self.transform:
            transf = transforms.ToTensor()
            image = transf(image)  # tensor数据格式是torch(C,H,W)
        return image,pose                                        #返回数据
