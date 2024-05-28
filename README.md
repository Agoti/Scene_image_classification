
# 人工智能原理 项目2

## 简介
利用深度学习方法解决风景图像分类的问题，最高准确率为0.87。

## 运行方法
1. 克隆仓库(我交的作业应该是一个空白仓库，可以不用运行git clone)
```bash
git clone https://github.com/Agoti/Scene_image_classification.git
cd Scene_image_classification
```
2. 安装依赖
```bash
conda create -n sic
conda activate sic
pip install -r requirements.txt
```
3. 运行启动脚本(我在最后一次commit的时候不小心把logs文件夹提交了，所以运行脚本的时候可能会针对logs文件夹报错，不用管)
```bash
bash scripts/setup.sh
```
3. 下载数据集  
数据集链接（可能会过期）:https://cloud.tsinghua.edu.cn/seafhttp/files/4bcb8510-de03-4dd6-856e-97f0a9beb3f9/scene_classification.zip
```bash
cd data
wget [清华云盘链接]
unzip scene_classification.zip
cd ..
```
4. 训练模型
```bash
bash scripts/train.sh
```
我训练的时候使用一张2080Ti的显卡，30个epoch大概需要15分钟。
> 训练脚本的使用  
> 可以参考`scripts/train.sh`和`scripts/experiments/*.sh`，它们是训练脚本的示例，包含很多命令行参数的设置，可以直接运行。  
> `scripts/train.sh`是目前模型的最好设置，包含了学习率调度，归一化，数据增强等优化方法。
> 模型的设置有两种方法，一种是改写config目录下的默认配置文件，另一种是在运行脚本时传入参数，命令行参数会覆盖配置文件的设置。  
> 请参考`Train.py`的构造函数，`Utils.py`和`Config.py`了解json配置文件的结构，作用和用法。  

如果不方便训练模型，我将一个checkpoint文件放在了云盘链接上，可以直接下载使用，大约130MB。请解压到`checkpoints`文件夹下。链接：https://cloud.tsinghua.edu.cn/d/17fd7cd1d4314a61bd7e/

5. 测试模型
```bash
bash scripts/test.sh
```
> 请参考Test.py的构造函数和main函数，了解命令行参数的设置和用法。Test.py的命令行参数比Train.py少很多。  
测试结束后会打印准确率，更多指标被存储在了相应`checkpoint`文件夹下的`metrics_test.json`文件中。  

指标：  
- overall：整体指标
  - 包含了accuracy，precision，recall，f1-score，confusion matrix, auc等指标，precision/recall/f1-score取所有类别的平均值。
- detailed：每个类别的指标
  - accuracy：准确率(二分类，是/不是每个类，所以数值很高)

6. 可视化Grad-CAM
```bash
python3 Visualize.py
```
Visualize.py需要一台有显示屏的设备，因为它会显示图片。

请修改Visualize.py中main函数中的checkpoint路径。如果你换了设备，或者移动了数据集的位置，可以在`[checkpoints]/config/dataset_config.json`中修改数据集的路径。  

`dataset_config.json`会针对Infinity报错，不用管，不影响使用。

我在做作业的时候，训练和测试在服务器上进行，device='cuda'，但是在本地可视化的时候，device='cpu'，所以在Visualize.py中，我将device设置为了'cpu'。

  
## 项目结构
- Config.py: 定义了配置文件的结构，包括了数据集参数，模型参数，训练参数等。
- Dataset.py: 定义了数据集的类，包括了数据集的加载，预处理，数据增强等。
- Model: 定义了模型的类，包括了前向传播，保存等。
  - Model.py: 定义了模型的接口，是所有模型的基类。
  - AlexNet.py: 定义了一些AlexNet模型。
- Train.py: 定义了训练的类，包括了训练的过程，训练的评估，训练的保存等。
- Test.py: 定义了测试的类，包括了测试的过程，测试的评估等。
- Metrics.py: 定义了评估指标的类，包括了准确率，混淆矩阵，AUC等。
- Utils.py: 定义了一些工具函数，包括了模型的构建，数据集的构建等。
- Visualize.py: 定义了Grad-CAM的计算和可视化。
- scripts: 定义了一些脚本，包括了训练，测试等。
- logs: 存储了训练的日志(控制台输出)
- checkpoints: 存储了训练的模型
- data: 存储了数据集
- config: 存储了配置文件
- requirements.txt: 存储了依赖的库
- README.md: 项目的说明文档
