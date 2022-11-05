# csicc

#### 介绍
c site image classify c(csicc)，use hash、cos、cnn to represent image that from c site home page.


#### 后续待测试场景
1. 选定一个模版网页，对该网页进行数据增强处理，衍生出n个图片，出来模型只针对该类网站识别；


#### 软件架构
该图像分类算法基于lenet和alexnet


#### 工程结构说明

+ csic 程序根目录
   + main  程序主要代码
       + classify 分类算法目录
       + config  存放程序配置参数目录
       + data_predict 数据预测代码
       + data_preprocess 数据预处理代码
       + model_test 模型测试代码
       + model_train 模型训练代码
       + utils 存放程序工具类目录
   + data  存放模型训练数据集目录
       + dataset  原始数据集目录
       + model 训练模型文件目录
       + predict 测试预测结果数据目录
   + doc 存放系统的文档目录
   - .gitignore 提交git需要忽略的配置文件
   - README.md  程序代码说明
   - requirements.txt  程序依赖的第三方包与版本
   - start.sh  程序启动脚本

#### 环境配置
(1) 创建虚拟环境
    conda create -n csic python==3.6.12
    pip install -r requirements.txt    # 在pycharm下面的Terminal即可执行
    
    提示：win10按照requirements版本安装，肯定是可正常运行的。
    
    注意：
    Mac安装后，python版本是3.6.13
    Win10安装后，python版本从3.6.0自动升级3.7.13，此时会有问题（不能升级到3.7，跑不了）
    （1）mkl-service package failed to import
    参考地址：https://www.cnblogs.com/ssjxx98/p/11222835.html
    （2）SystemError: error return without exception set  
    参考地址：https://blog.csdn.net/weixin_44189688/article/details/106122021

    conda升级命令：conda update -n base -c defaults conda
    按指定安装依赖包参考命令如下：
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.13.1
    
    keras与tensorflow版本对应关系参考如下：
    https://blog.csdn.net/weixin_43590232/article/details/108239114
    设置conda清华源镜像
    https://blog.csdn.net/jialong_chen/article/details/121485308
    .condarc配置参考如下，注意http，不是https
    ssl_verify: true
    show_channel_urls: true
    channel_alias: http://mirrors.tuna.tsinghua.edu.cn/anaconda
    default_channels:
      - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
      - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
      - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
      - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
      - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
      
    matplotlib显示中文乱码解决字体包：data/fonts/SimHei.ttf
    
    经验：
    1、tensorflow-1.13.1对应的CUDA-10.0，即CUDA Toolkit-10.0.130,Windows驱动>=411.31,Linux驱动>=410.38
    2、使用GPU训练模型比CPU要快的多。
    3、[工程中的哈希相似度代码参考地址](https://blog.csdn.net/haofan_/article/details/77097473)
	4、[keras steps_per_epoch=sample_numbers/batch_size](https://blog.csdn.net/qq_36336522/article/details/103120167)
	5、channels（颜色通道），结合图片表达可以分为两种，即channels_first（3,224,224）、channels_last（224,224,3），通道
	的不同位置会对训练模型性能有影响。keras默认是在后面，即channels_last（224,224,3）。
	6、win10安装GraphViz参考命令：conda install GraphViz --channel conda-forge -y
	7、激活函数（例如Relu、Sigmoid）与激活层（Activation）不是
	8、[tqdm是一个进度条依赖包](https://zhuanlan.zhihu.com/p/163613814)
	
	Dense参数解释
	    units: 该层有几个神经元
	    activation：该层使用的激活函数
	    use_bias：是否使用偏置
	    kernel_initializer：权重的初始方法
	    bias_initializer：偏置的初始化方法
	    kernel_regularizer：权重的规范化函数
	    bias_regularizer：偏置的规范化函数
	    activity_regularizer：输出的规范化函数
	    kernel_constraint：权重的限制函数
	    bias_constraint：偏置的限制函数
	    input_dim：输入参数维度（1维数组、2维数组）
	    input_shape：输入参数矩阵，例如(2,3)
    
1. [版本问题---keras和tensorflow的版本对应关系](https://www.cnblogs.com/carle-09/p/11661261.html)
2. [Keras——检查GPU是否可用](https://blog.csdn.net/qq_33182424/article/details/106080243) 
3. [英伟达CUDA驱动下载地址](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)  
4. [ubuntu 16.04 使用keras框架 对比cpu和gpu训练神经网络的速度](https://www.likecs.com/show-204281916.html)
5. [CPU和GPU跑深度学习差别](https://www.zhihu.com/question/273812506/answer/2318336356)
6. [dense层参数及理解](https://blog.csdn.net/orDream/article/details/106355491)
7. [](https://blog.csdn.net/nyist_yangguang/article/details/121630044)
7. [keras绘图工具graphviz](https://www.graphviz.org/)
8. [win10安装graphviz方法](https://blog.csdn.net/fuge92/article/details/88371693)
    

(2) 直接拷贝虚拟环境
    windows版将tensorflow复制到本地anaconda目录下的envs中解压即可使用
    linux版将py36复制到/root/anaconda3/envs下解压即可使用
 
 #### 使用说明   
1. 模型训练 
    Python main.py
    在模型训练时，需要修改config.py 文件中的模型命名及保存路径
    
    使用keras基本流程方法：
        1、[导入keras] ——> [定义模型] ——> [编译模型] 
        2、[训练模型] ——> [评估模型] ——> [保存模型]

2. main.py 用于训练模型
    将训练所需要的数据放入dataset/CT_X/[target/...|other/...]
    
3. 参数设置在 config.py 文件中
    
   1. 模型预测
       预测： 
           python predict_single.py 单分类预测
           python predict_all.py 全部分类预测，由于部分类别数据量较少，部分的识别由深度学习与相似度共同完成
               推断数据是否需要从minio下载并且是否上传至minio由config.py中的download_minio|upload_minio两部分控制
       a. 在使用predict_all.py 时， 需要对config.py 中的sampleTotalNum进行修改
       b. 需要对存储的模型进行重命名 [ss_model_03-weights.best.hdf5] '03' 代表第三类模板识别模型
       c. 需要对config.py 中的 [percents, use_sim, images] 进行修改
    
4. 本项目中包含在数据收集过程中通过进行图像相似度计算来辅助图像审核工作
    图像相似度包含两种算法
        一：余弦相似度
            通过计算两个向量的夹角余弦值来评估他们的相似度
        二：均值哈希
            将图片缩小尺寸后计算所有像素点的灰度平均值，将每个像素的灰度，与平均值进行比较。大于或等于平均值，记为1；
            小于平均值，记为0。计算两组64位数据的汉明距离，即对比数据不同的位数，不同位数越少，表明图片的相似度越大,
            均值哈希算法计算速度快，不受图片尺寸大小的影响

5. auto_train使用 classes类别数
    模型自动训练时，将收集的数据放在指定文件夹中，文件夹命名方式CT_classes
    python auto_train.py --model lenet --classes 18 --acc 0.9 注[model值本模板采用模型的类型 classes指第几种模板 acc指期望准确率]
    模型将进行训练，训练完成后如果阈值未达到期望准确率，将不保存模型，若达到，将保存，并放在相关文件夹中