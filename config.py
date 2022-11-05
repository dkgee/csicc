# coding: UTF-8
import os
import time
# from data_process import get_picture_name

root_dir = os.path.dirname(__file__)


# 获取图片文件名称
def get_picture_name(file_path):
    for _, _, files in os.walk(file_path):
        pass
    return files


class Config(object):
    def __init__(self):
        """
        train
        """
        # 训练模型需要进行修改
        # self.model_path = os.path.join(root_dir, 'data/save_model/lenet_model_11-weights.best.hdf5')  # 未训练
        self.model_path = os.path.join(root_dir, 'data/model/lenet_model_01-weights.best.hdf5')  # 未训练
        self.train_for_saved_model_path = os.path.join(root_dir,
                                                       'data/save_model/12_31_sample_model_10-weights.best.hdf5')
        # 训练数据集所在目录
        # self.train_path = os.path.join(root_dir, 'dataset/CT_18/train')
        self.train_path = os.path.join(root_dir, 'data/dataset/ct_01/train')
        # 测试数据集所在目录 TODO 此处需要改成验证数据集（参考main.py 第63行）
        # self.test_path = os.path.join(root_dir, 'dataset/CT_11/test')
        self.test_path = os.path.join(root_dir, 'data/dataset/ct_01/test')
        # 验证数据集所在目录
        self.valid_path = os.path.join(root_dir, 'dataset/ct_01/valid')
        # train
        self.epoch = 100
        # 50、65、80、100
        # self.epoch = 50
        self.lr = 0.0005
        # 训练的周期数steps_per_epoch=sample_numbers / batch_size，以训练CT_01为例，共计1666个样本，每批取64个，共计26个批次，批次数量越大，占用内存越大
        self.steps_per_epoch = 24
        self.resize = 224
        self.batch_size = 64
        # TODO 图片保存路径
        self.model_num = len(get_picture_name(os.path.join(root_dir, "data/save_model")))
        self.model01params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.5, "l2": 0.01}
        self.model02params = {"lr": 0.00001, "decay": 1e-5, "momentum": 0.8, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model03params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.8, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model04params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model05params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.8, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model06params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.8, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model07params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.8, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model08params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.95, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model09params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.95, "loss": "categorical_crossentropy",
                              "dropout": 0.7, "l1": 0.01}
        self.model10params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.7, "l1": 0.01}
        self.model11params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.7, "l1": 0.01}
        self.model12params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model13params = {"lr": 0.001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model14params = {"lr": 0.0001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model15params = {"lr": 0.001, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01}
        self.model16params = {"lr": 0.005, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01, "patience": 50}
        self.model17params = {"lr": 0.0005, "decay": 1e-5, "momentum": 0.9, "loss": "categorical_crossentropy",
                              "dropout": 0.3, "l1": 0.01, "patience": 50}

        """
        predict
        """
        # 模型预测文件地址 需要时二级目录例如：train/all/**.png
        self.predict_dir_path = os.path.join(root_dir, 'data/test_ss')

        # ## predict_single
        # 加载模型的地址
        self.model = os.path.join(root_dir, "data/save_model/ss_model_05-weights.best.hdf5")
        self.num = 11
        self.predict_model_path = os.path.join(root_dir, "data/model/lenet_model_17-weights.best.hdf5")
        # if self.num <= 9:
        #     self.predict_model_path = os.path.join(root_dir, 'data/save_model/ss_model_0{}-weights.best.hdf5'.format(
        #         self.num))
        # else:
        #     self.predict_model_path = os.path.join(root_dir, 'data/save_model/ss_model_{}-weights.best.hdf5'.format(
        #         self.num))
        # self.predict_model_path = os.path.join(root_dir, "data/save_model/12_30_sample_model_10-weights.best.hdf5")
        # 模型预测设置阈值
        self.model_percent = 0.9
        # 模型预测是否需要相似度辅助
        self.use_sim = False
        # 模型相似度设置阈值
        self.sim_percent = 0.9
        self.img = os.path.join(root_dir, "data/ss_sample/{}.png".format(self.num))
        # self.img = os.path.join(root_dir, "data/ss_sample/5.png")
        self.hash = "aHash"
        # 需要预测的本地文件 30 28 23 15 08  #wrong, right, notfound 2, 5, 1|5, 25, 7|3, 11, 2|2, 5, 0
        self.predicted_folder = "E:/data/unzip/20220218_22"  # 0.95
        # self.predicted_folder = "./data/predict_dir/ss_sample_05"
        # ss文件保存地址
        self.predictSSpath = os.path.join(root_dir, "data/predict_dir/ss_sample_18")
        # 其他文件保存地址
        self.predictUNKpath_s = os.path.join(root_dir, "data/predict_dir/ss_sample_19")

        # ## predict_all
        # 需要识别的模板数量
        self.sampleTotalNum = 17
        # 模型识别过程中是否使用相似度算法、概率阈值及相似度识别概率阈值 0.86
        # [1用|0不用, model_percent, sim_percent, path]
        self.percents = 0
        self.predictUNKpath = os.path.join(root_dir, "data/predict_dir/predict_other")
        # [use pic sim, model_percent, sim_percent, hash_type, use_text_filter]
        self.predict_params = {"per_01": [1, 0.981, 0.6, os.path.join(root_dir, "data/ss_sample/1.png"), 'adHash', 0],
                               "per_02": [1, 0.98, 0.6, os.path.join(root_dir, "data/ss_sample/2.png"), 'adHash', 0],
                               "per_03": [0, 0.95, 0.8, os.path.join(root_dir, "data/ss_sample/3.png"), 'adHash', 0],
                               "per_04": [1, 1, 0.9, os.path.join(root_dir, "data/ss_sample/4.png"), 'adHash', 0],
                               "per_05": [1, 0.7, 0.5, os.path.join(root_dir, "data/ss_sample/5.png"), 'pmHash', 1],
                               "per_06": [1, 0.6, 0.1, os.path.join(root_dir, "data/ss_sample/6.png"), 'adHash', 1],
                               "per_07": [0, 0.9, 0.8, os.path.join(root_dir, "data/ss_sample/7.png"), 'adHash', 0],
                               "per_08": [1, 0.9, 1, os.path.join(root_dir, "data/ss_sample/8.png"), 'adHash', 0],
                               "per_09": [0, 0.999, 0.8, os.path.join(root_dir, "data/ss_sample/9.png"), 'adHash', 0],
                               "per_10": [1, 0.999, 0.8, os.path.join(root_dir, "data/ss_sample/10.png"), 'adHash', 0],
                               "per_11": [1, 0.8, 0.6, os.path.join(root_dir, "data/ss_sample/11.png"), 'adHash', 1],
                               "per_12": [1, 0.998, 0.7, os.path.join(root_dir, "data/ss_sample/12.png"), 'adHash', 0],
                               "per_13": [1, 0.999, 0.82, os.path.join(root_dir, "data/ss_sample/13.png"), 'adHash', 0],
                               "per_14": [1, 0.9, 0.9, os.path.join(root_dir, "data/ss_sample/14.png"), 'adHash', 0],
                               "per_15": [1, 0.9, 0.9, os.path.join(root_dir, "data/ss_sample/15.png"), 'adHash', 0],
                               "per_16": [1, 0.9, 0.95, os.path.join(root_dir, "data/ss_sample/16.png"), 'adHash', 0],
                               "per_17": [1, 0.98, 0.75, os.path.join(root_dir, "data/ss_sample/17.png"), 'adHash', 0]}

        self.key_text = {
            "per_05": "Powered by SSPANEL Theme by editXY",
            "per_06": "Copyright",
            "per_11": "POWERED BY SSPANEL"
        }

        """
        similarity
        """
        self.resize_sim = 64
        self.img_sim = os.path.join(root_dir, "data/ss_sample/hb-1.png")
        """
        minio
        """
        datetime = time.strftime("%Y%m%d", time.localtime())
        # datetime = '20220108'
        # 下载bucket
        self.download_bucket = 'test'
        # 下载二级目录
        self.download_bucket_2 = datetime
        # 上传bucket
        self.upload_bucket = 'testzl'
        self.local_dir_2 = 'all'
        self.download_minio = True
        self.upload_minio = True
        self.download_to_path = os.path.join(root_dir, "dataset/predicted")


if __name__ == "__main__":
    con = Config()
