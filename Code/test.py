import onnxruntime as ort
import numpy as np
import torch, cv2
import torchvision.transforms as transforms 


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preict_one_img(img_path):
    image = cv2.imread(img_path)                           #读取图像数据
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image, 0).repeat(10,axis=0)
    image = image.astype(np.float32)
    # transf = transforms.ToTensor()
    # image = transf(image)  # tensor数据格式是torch(C,H,W)
    # image=image.unsqueeze(0)
    # print(image.shape)
    # outputs = ort_session.run(
    # None,
    # {"input.1":torch.rand(10, 3, 360, 500)},
    # )
    # np.random.randn(10, 3, 360, 500).astype(np.float32)
    outputs = ort_session.run(
    None,
    {"input.1":image},
    )
    print(outputs)




if __name__ == '__main__':
    # classes = models.AlexNet_Weights.IMAGENET1K_V1.value.meta["categories"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = "./1.jpg"
    ort_session = ort.InferenceSession("lenet.onnx", providers=['CUDAExecutionProvider'])
    # print("----------------- 输入部分 -----------------")
    # input_tensors = ort_session.get_inputs()  # 该 API 会返回列表
    # for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表
    #     input_info = {
    #         "name" : input_tensor.name,
    #         "type" : input_tensor.type,
    #         "shape": input_tensor.shape,
    #     }
    #     print(input_info)
    preict_one_img(img_path)