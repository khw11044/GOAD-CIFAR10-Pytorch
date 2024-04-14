import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import torchvision.datasets as dset


def norm(data, mu=1):
    return 2 * (data / 255.) - mu

def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


root = './data'
class_ind = 0


testset = dset.CIFAR10(root, train=False, download=True)
test_data = np.array(testset.data)
test_labels = np.array(testset.targets)

classes = testset.classes

x_test = norm(np.asarray(test_data, dtype='float32'))
x_test_trans = x_test.transpose(0, 3, 1, 2)
y_test = (np.array(test_labels) == class_ind)

x_test_trans = x_test_trans.reshape(-1,8,3,32,32)
test_labels = test_labels.reshape(-1,8)
# 학습용 이미지 뽑기
dataiter = iter(x_test_trans)
# images, labels = next(dataiter)
images = next(dataiter)

# 이미지 보여주기
images_tensor = torch.from_numpy(images)
imshow(torchvision.utils.make_grid(images_tensor))
print()
# 이미지별 라벨 (클래스) 보여주기
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))