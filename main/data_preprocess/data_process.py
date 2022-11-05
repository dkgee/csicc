import os
from PIL import Image
import tqdm
from config import Config
from numpy import average, dot, linalg
import time
import cv2
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import re
import numpy as np
current_time = time.strftime("%Y-%m-%d")
config = Config()


# 获取图片文件名称
def get_picture_name(file_path):
    for _, _, files in os.walk(file_path):
        pass
    return files


# 复制图片到指定路径
def copy_picture_to_folder(old_path, new_path, pic_name, percent, template_num):
    shutil.copyfile(old_path, os.path.join(new_path, '{}_{}_{}'.format(template_num, percent, pic_name)))


def copy_picture_to_folder_sim(old_path, new_path, pic_name, percent, per_sim):
    shutil.copyfile(old_path, os.path.join(new_path, '{}_{}_{}'.format(round(per_sim, 3), round(percent, 3), pic_name)))


def copy_picture_to_folder_s(old_path, new_path, pic_name):
    shutil.copyfile(os.path.join(old_path, pic_name), os.path.join(new_path, pic_name))


# 对训练、测试数据集进行收集
def copy_train_test_pic(data_path, train_data_path, test_data_path, percent):
    pic_name = get_picture_name(data_path)
    random.shuffle(pic_name)
    if percent < 1:
        total_num = len(pic_name)
        num_1 = int(total_num * percent)
        num_2 = total_num - num_1
    else:
        num_1 = percent
        num_2 = percent // 3
    pic_name_1 = pic_name[:num_1]
    pic_name_2 = pic_name[num_1:num_1 + num_2]
    for j in tqdm.tqdm(pic_name_1, total=len(pic_name_1)):
        copy_picture_to_folder_s(data_path, train_data_path, j)
    for i in tqdm.tqdm(pic_name_2, total=len(pic_name_2)):
        copy_picture_to_folder_s(data_path, test_data_path, i)


# 图片镜像
def ImageMirror(pic_name, path, savePath):
    filepath = os.path.join(path, pic_name)
    img = Image.open(filepath)
    img_pixel = img.load()
    mirror = Image.new(img.model, img.size, "white")

    width, height = img.size
    """水平镜像转换，遍历每个像素点，将后列变前列"""
    for y in range(height):
        for x in range(width):
            pixel = img_pixel[width-1-x, y]
            mirror.putpixel((x, y), pixel)
    mirror.save(os.path.join(savePath, 'mirror_{}'.format(pic_name)))


# 旋转图片
def rotate_pic(pic_name, path, savePath):
    # 读取图像
    pic_path = os.path.join(path, pic_name)
    im = Image.open(pic_path)
    # im.show()
    # 指定逆时针旋转的角度
    im_rotate = im.rotate(180)
    # 展示图片
    # im_rotate.show()
    im_rotate.save(os.path.join(savePath, 'rotate_{}'.format(pic_name)))


# 判断图片数据是否可用
def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid


# 删除无效数据
def drop_invaild_data(folder_path):
    num = 0
    for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in tqdm.tqdm(os.listdir(sub_folder_path), total=len(os.listdir(sub_folder_path))):
            file_path = os.path.join(sub_folder_path, filee)
            if is_valid(file_path):
                from PIL import ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                Image.MAX_IMAGE_PIXELS = None
            else:
                os.remove(file_path)
                num += 1
    return num


def drop_invaild_data_simple(folder_path):
    num = 0
    for filee in tqdm.tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, filee)
        if is_valid(file_path):
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            Image.MAX_IMAGE_PIXELS = None
        else:
            os.remove(file_path)
            num += 1
    return num


# 删除已经预测过的原始文件中的数据
# 在业务中，每天数据不定期进行更新，模型配置中按小时来进行定期执行，对已经执行过的数据进行删除操作，避免重复审核
def del_files(path_file, islist=False):
    if islist:
        for pf in path_file:
            ls = os.listdir(pf)
            for i in ls:
                f_path = os.path.join(pf, i)
                # 判断是否是一个目录,若是,则递归删除
                if os.path.isdir(f_path):
                    del_files(f_path)
                else:
                    os.remove(f_path)
    else:
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                del_files(f_path)
            else:
                os.remove(f_path)


""" 图片数据处理 """


# 对图片进行统一化处理
def get_thum(image, size=(8, 8), greyscale=True):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


# 计算图片之间的相似度
def calculated_similarity(source_dir, target_dir, other_dir, pic_names, img1):
    num = 0
    for i in tqdm.tqdm(pic_names, total=len(pic_names)):
        img2 = Image.open(os.path.join(source_dir, i))
        sim = image_similarity_vectors_via_numpy(img1, img2)

        if sim > config.percent:
            copy_picture_to_folder(source_dir, target_dir, i, "{}_{}_{}".format(sim, current_time, i))
            num += 1
        else:
            # pass
            copy_picture_to_folder(source_dir, other_dir, i, "{}_{}_{}".format(sim, current_time, i))
    return num


# hash distance
# 均值哈希算法
def aHash(img, resize):
    # 缩放为8*8
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(resize):
        for j in range(resize):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / (resize*resize)
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(resize):
        for j in range(resize):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值感知算法
def dHash(img, resize):
    # 缩放8*8
    img = cv2.resize(img, (resize+1, resize), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(resize):
        for j in range(resize):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str


def flatten(x):
    result = []
    for el in x:
        result.extend(el)
    return result


def pHash(img, resize):
    """get image pHash value"""
    # 加载并调整图片为8*8灰度图片
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img       # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(resize, resize)

    # 把二维list变成一维list
    img_list = flatten(vis1.tolist())

    # 计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]
    hash_str = ''
    for i in avg_list:
        hash_str += str(i)
    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]), 2) for x in range(0, resize*resize)])


def p_middle_Hash(imgfile):
    """get image pHash value"""
    # 加载并调整图片为32x32灰度图片
    # img = cv2.imread(imgfile, cv2.INTER_CUBIC)
    # 加载并调整图片为8*8灰度图片

    img = cv2.resize(imgfile, (32, 32), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # 通过人为设定侧重区域来实现相似度算法注意点
    vis1 = vis1[0:8, 12:20]  # 取矩阵中间8*8数据
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    # vis1.resize(8, 8)

    # 把二维list变成一维list
    img_list = flatten(vis1.tolist())

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 64)])


def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    s = sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
    return s


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n+1
    return n


def hash_similarity(img1, img2, hash, resize):
    resize2 = resize * resize
    if hash == "aHash":
        # 均值哈希
        hash1 = aHash(img1, resize)
        hash2 = aHash(img2, resize)
        n = cmpHash(hash1, hash2)
        percent = (resize2 - n) / resize2
    elif hash == "dHash":
        # 差值哈希
        hash1 = dHash(img1, resize)
        hash2 = dHash(img2, resize)
        m = cmpHash(hash1, hash2)
        percent = (resize2 - m) / resize2
    elif hash == "pHash":
        # 感知哈希
        hash1 = pHash(img1, resize)
        hash2 = pHash(img2, resize)
        p = hammingDist(hash1, hash2)
        percent = (resize2-p)/resize2
    elif hash == "pmHash":
        # 感知哈希
        hash1 = p_middle_Hash(img1)
        hash2 = p_middle_Hash(img2)
        p = hammingDist(hash1, hash2)
        percent = (64 - p) / 64
    else:  # 差值和平均的平均
        # 均值哈希
        hash1 = aHash(img1, resize)
        hash2 = aHash(img2, resize)
        n = cmpHash(hash1, hash2)
        precent_a = (resize2 - n) / resize2
        # 差值哈希
        hash1 = dHash(img1, resize)
        hash2 = dHash(img2, resize)
        m = cmpHash(hash1, hash2)
        precent_d = (resize2 - m) / resize2
        percent = (precent_d+precent_a)/2

    return percent


# 获取相似图片
def get_sim_pic(params, img2path, resize, cos=True):
    img1path = params[3]
    if cos:
        img1 = Image.open(img1path)
    else:
        img1 = cv2.imread(img1path)

    if not cos:
        img2 = cv2.imread(img2path)
        pre = hash_similarity(img1, img2, params[4], resize)
    else:
        img2 = Image.open(img2path)
        pre = image_similarity_vectors_via_numpy(img1, img2)
    return pre


# 获取相似图片
def get_sim_pic_s(img1path, img2path, resize, cos=True):
    if cos:
        img1 = Image.open(img1path)
    else:
        img1 = cv2.imread(img1path)

    if not cos:
        img2 = cv2.imread(img2path)
        pre = hash_similarity(img1, img2, config.hash, resize)
    else:
        img2 = Image.open(img2path)
        pre = image_similarity_vectors_via_numpy(img1, img2)
    return pre


# 将数据转换为list
def to_list(predicted_class_indices):
    l = str(predicted_class_indices)
    l = l.replace('[', '').replace(']', '')
    m = []
    label_0 = 0
    label_1 = 0
    for j in l.split(' '):
        if j == "0":
            m.append(0)
            label_0 += 1
        else:
            m.append(1)
            label_1 += 1
    return m


# 获取真实的list标签（使用label在filenames搜索，检查文件名是否包含标签）
def get_true_label(filenames, label):
    true_label = []
    try:
        for j in filenames:
            if re.search(label[0], j):
                true_label.append(0)
            else:
                true_label.append(1)
    except KeyError as e:
        print(e)
    return true_label


def get_predict_label(predict_list, percent):
    pre_list = []
    for i in predict_list:
        if i[1] > percent:
            pre_list.append(1)
        else:
            pre_list.append(0)
    return pre_list


def model_predict_ss(filenames, pred, percent):
    names = []
    for idx, name in zip(range(len(filenames)), filenames):
        name = str(name)
        name = name.split('\\')[1]
        if pred[idx][1] > percent:
            names.append(name)
    return names


def similarity_predict_ss(pic_dir, unk_pic_names, ss_key):
    names = []
    for unk_name in unk_pic_names:
        pic_path = os.path.join(pic_dir, unk_name)
        sim_pre = get_sim_pic(params=config.predict_params[ss_key], img2path=pic_path,
                              resize=config.resize_sim, cos=False)
        if sim_pre > config.predict_params[ss_key][2]:
            names.append(unk_name)

    return names


def img_process(predict_dir_path):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(predict_dir_path,
                                                      target_size=(224, 224), batch_size=1,
                                                      class_mode='categorical', shuffle=False)
    test_generator.reset()
    return test_generator


def predict_pic(model_path, test_generator):
    model = load_model(model_path)

    # 预测结果
    pred = model.predict_generator(test_generator, verbose=0, steps=len(test_generator))
    return pred


def text_filter(df, pic_name, key_text):
    # df = pd.read_csv(path, encoding='gb18030')
    try:
        ind = df.query("screenshot_name=='{}'".format(pic_name)).index[0]
    except:
        ind = None
    if ind != None:
        text = df['html_text'][ind]
        if len(str(text)) == 0:
            text = ''
    else:
        text = ''
    if key_text in str(text):
        return 1
    else:
        return 0

