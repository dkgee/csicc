

import os
import cv2
from main.classify.hash.hashing import pic_gray


def demo01():
    """
    图片灰度处理过程演示
    """
    root_dir = os.path.dirname(__file__)
    # sample_path = os.path.join(root_dir, '../../../data/dataset/cosin/sample/6643956b38be58b365afa13221164c20.png')
    # gray_path = os.path.join(root_dir, '../../../data/dataset/cosin/sample/6643956b38be58b365afa13221164c20-gray.png')
    sample_path = os.path.join(root_dir, '../../../data/dataset02/train/c_01/0f543322c9e228297f49b974842bbc60.png')
    gray_path = os.path.join(root_dir, '../../../data/dataset02/train/c_01/0f543322c9e228297f49b974842bbc60-gray-8.png')
    img1 = cv2.imread(sample_path)
    # resize = 64 # 生成的哈希值太大，长度为4096
    resize = 8
    pic_gray(img1, resize, gray_path)


if __name__ == '__main__':
    demo01()
