from unittest import case
import torch
from torchvision import datasets, transforms
from random import randint
import numpy as np
from PIL import Image

class MNIST_Test():
    def __init__(self):
        self.test_batch_size = 100
        self.load_test()

    def load_test(self):
        self.test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/mnist/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=self.test_batch_size,
        shuffle=True,
        num_workers=1,
        timeout=600)
    
    def get_random_testcase(self):
        data, target = next(iter(self.test_loader))
        case_num = randint(0, len(data) - 1)
        z = data.numpy()[case_num].astype(np.float32)
        test_case = data.numpy()[case_num].ravel().astype(np.float32)
        test_name = target.numpy()[case_num]
        return test_case, test_name
    
    # def show_test_case(self, test_case):
    #     img = test_case.copy()
    #     img = img.reshape((28, 28))
    #     print(img.shape)
    #     img=Image.fromarray(img, "L")    
    #     img.show()
