# 导入依赖包
import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddleslim.auto_compression import AutoCompression

paddle.enable_static()


# 定义DataSet
class ImageNetDataset(DatasetFolder):
    def __init__(self, path, image_size=224):
        super(ImageNetDataset, self).__init__(path)
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size), transforms.Transpose(),
            normalize
        ])

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.samples)


# 定义DataLoader
train_dataset = ImageNetDataset("./ILSVRC2012_data_demo/ILSVRC2012/train/")
image = paddle.static.data(
    name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
train_loader = paddle.io.DataLoader(train_dataset, feed_list=[image], batch_size=32, return_list=False)
# 开始自动压缩
ac = AutoCompression(
    model_dir="./MobileNetV1_infer",
    model_filename="inference.pdmodel",
    params_filename="inference.pdiparams",
    save_dir="MobileNetV1_quant",
    # config={"QuantPost": {}, "HyperParameterOptimization": {'ptq_algo': ['avg'], 'max_quant_count': 3}},
    config={"QuantAware": {}, "Distillation": {}},  ### 如果您的系统为Windows系统, 请使用当前这一行配置
    train_dataloader=train_loader,
    eval_dataloader=train_loader,
)
ac.compress()
