# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(2, batch_size)
print(batch_mask)

# img = x_train[0]
# label = t_train[0]
# img = img.reshape(28, 28)  # 形状を元の画像サイズに変形

# img_show(img)

# print(x_test.shape)
print("\n")
