
import os
import cv2
from main.classify.hash.hashing import hash_similarity


# 计算两个图片的哈希值的相似度
def cal_pic_sim_by_hash(img1_path, img2_path, hash_type, resize):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    pre = hash_similarity(img1, img2, hash_type, resize)
    return pre
