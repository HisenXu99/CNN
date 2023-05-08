import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

#import cv2

cam0_path  = '/home/hisen/Project/Data/CR5_picture/'    # 已经建立好的存储cam0 文件的目录


def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    dir_path = Path(dir_path)
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表
    name_list= []

    for img_path in dir_path.glob('*.jpg'):
        images_list.append(str(img_path))
    #sorted(path, key = os.path.getctime)
    with open(cam0_path+'test.txt','r',encoding = 'utf-8') as f:
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
    print(name_list)
    return images_list, labels_list


def main():
    name,list=get_images_and_labels(cam0_path)
    print(name)
    print(list)


if __name__ == '__main__':
    main()
