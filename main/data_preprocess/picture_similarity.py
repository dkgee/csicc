import os
import tqdm
from config import Config
from main.data_preprocess.data_process import get_sim_pic_s
from main.data_preprocess.data_process import get_picture_name, del_files, copy_picture_to_folder
import time
s = time.time()


config = Config()
unk_dir = 'E:/data/unzip/20220209_14/all'
# unk_dir = os.path.join(config.predicted_folder, config.local_dir_2)
del_files(config.predictSSpath)
del_files(config.predictUNKpath_s)

unk_pic_names = get_picture_name(unk_dir)
n = 0
for i in tqdm.tqdm(unk_pic_names, total=len(unk_pic_names)):
    pic2path = os.path.join(unk_dir, i)
    percent = get_sim_pic_s(config.img_sim, pic2path, config.resize_sim, cos=False)
    # print(percent)
    if percent > config.sim_percent:
        copy_picture_to_folder(pic2path, config.predictSSpath, i, percent)
        n += 1
    else:
        copy_picture_to_folder(pic2path, config.predictUNKpath_s, i, percent)
print('total number:', n)


def cacu(other_dir, ss_dir):
    pre_other_name = get_picture_name(config.predictUNKpath_s)

    pre_ss_name = get_picture_name(config.predictSSpath)
    print(pre_ss_name)
    other_name = get_picture_name(other_dir)
    ss_name = get_picture_name(ss_dir)

    other_true = 0
    other_false = 0

    ss_true = 0
    ss_false = 0

    for other in pre_other_name:
        if other in other_name:
            other_true += 1
        else:
            other_false += 1
    for ss in pre_ss_name:
        if ss in ss_name:
            ss_true += 1
        else:
            ss_false += 1
    print(other_true, ss_true, other_false, ss_false)
    precision = ss_true/(ss_true+ss_false)
    recall = ss_true/(ss_true+other_false)
    # print("fp", other_true, "other false", other_false)
    # print("ss_treu", ss_true, "ss_false", ss_false)
    print("precision", ss_true/(ss_true+ss_false))
    print("acc", (ss_true + other_true)/(other_true+other_false+ss_true+ss_false))
    print("recall", ss_true/(ss_true+other_false))
    print("F1", 2*precision*recall/(precision+recall))


# other = 'data/test_ss/other'
# ss = 'data/test_ss/ss'
# cacu(other, ss)
e = time.time()
print(e-s)



# percent = get_sim_pic(config.img, pic2path, cos=False)

