import numpy as np
import random
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F


import matplotlib.cm as cm


def make_dataset(image_list, land, biocular, au):
    len_ = len(image_list)
    images = [(image_list[i].strip(), land[i, :], biocular[i], au[i, :]) for i in range(len_)]

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def add_noise(heatmap, channel, label, center_1, center_2, sigma, size, threshold=0.01):
    # 生成一个大小为 (size, size) 的二维高斯权重矩阵 gauss_noise_1
    # 计算公式：
    # G(x, y) = ((x - x0)^2 + (y - y0)^2) / (-2 * sigma^2)
    # 其中：
    # - (x0, y0) = center_1 是高斯分布的中心
    # - sigma 控制高斯分布的扩散程度（标准差）
    # - np.fromfunction() 逐像素计算高斯加权值
	gauss_noise_1 = np.fromfunction(lambda y,x : ((x-center_1[0])**2 \
											+ (y-center_1[1])**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
	gauss_noise_1 = np.exp(gauss_noise_1)
	gauss_noise_1[gauss_noise_1 < threshold] = 0
	gauss_noise_1[gauss_noise_1 > 1] = 1
	gauss_noise_2 = np.fromfunction(lambda y,x : ((x-center_2[0])**2 \
											+ (y-center_2[1])**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
    # 标签主导的响应叠加
	if label[channel] == 1:
		heatmap[channel] += gauss_noise_1 # AU存在时增强
	else:
		heatmap[channel] -= gauss_noise_1 # AU不存在时抑制
          
    # 设计意义：
    # 某些 AU 需 双区域联动触发（例如同时检测左右嘴角）
    # 当只需要单区域响应时，通过 center_2[0] = -1 规避该计算
	if center_2[0] == -1:   # 第二个点无效时退出
		return heatmap

	gauss_noise_2 = np.exp(gauss_noise_2)
	gauss_noise_2[gauss_noise_2 < threshold] = 0
	gauss_noise_2[gauss_noise_2 > 1] = 1
	if label[channel] == 1:
		heatmap[channel] += gauss_noise_2
	else:
		heatmap[channel] -= gauss_noise_2
	return heatmap

def heatmap2au(heatmap):
	avg = torch.mean(heatmap, dim=(2,3))
	label = (avg > 0).int()

	return label


# 用于将面部关键点 (landmark) 转换为热力图层 (heatmap)，主要用于面部动作编码 (Facial Action Unit, AU) 检测或相关视觉任务
def au2heatmap(lmk, label, size, sigma, config):
    
    #将关键点坐标转换为 [N, 2] 的二维坐标
    lmk = lmk.reshape(len(lmk)//2, 2)

    if size == 256:
        lmk = lmk
    elif size == 128:
        lmk = lmk/2 # [68, 2] -> 49
    elif size == 64:
        lmk = lmk/4 # [68, 2] -> 49
    else:
        raise TypeError
    
    # 热图初始化
    heatmap = np.zeros((config.MODEL.NUM_CLASSES, size, size))
    lmk_eye_left = lmk[19:25]   # 左眼关键点
    lmk_eye_right = lmk[25:31]  # 右眼关键点
    eye_left = np.mean(lmk_eye_left, axis=0)    # 左眼中心坐标
    eye_right = np.mean(lmk_eye_right, axis=0)  # 右眼中心坐标
    lmk_eyebrow_left = lmk[0:5]     # 左眉关键点
    lmk_eyebrow_right = lmk[5:10]   # 右眉关键点
    eyebrow_left = np.mean(lmk_eyebrow_left, axis=0)
    eyebrow_right = np.mean(lmk_eyebrow_right, axis=0)
    IOD = np.linalg.norm(lmk[25] - lmk[22])   # 瞳孔间距离

    if config.DATA.DATASET == 'BP4D':
        # au1 lmk 21, 22
        heatmap = add_noise(heatmap, 0, label, lmk[4], lmk[5], sigma, size)

        # au2 lmk 17, 26
        heatmap = add_noise(heatmap, 1, label, lmk[0], lmk[9], sigma, size)

        # au4 brow center
        heatmap = add_noise(heatmap, 2, label, eyebrow_left, eyebrow_right, sigma, size)

        # au6 1 scale below eye bottom
        heatmap = add_noise(heatmap, 3, label, [eye_left[0], eye_left[1]+IOD], [eye_right[0], eye_right[1]+IOD], sigma, size)

        # au7 lmk 38, 43
        heatmap = add_noise(heatmap, 4, label, lmk[21], lmk[26], sigma, size)

        # au10 lmk 50, 52
        heatmap = add_noise(heatmap, 5, label, lmk[33], lmk[35], sigma, size)

        # au12 lmk 48, 54
        heatmap = add_noise(heatmap, 6, label, lmk[31], lmk[37], sigma, size)

        # au14 lmk 48, 54
        heatmap = add_noise(heatmap, 7, label, lmk[31], lmk[37], sigma, size)

        # au15 lmk 48, 54
        heatmap = add_noise(heatmap, 8, label, lmk[31], lmk[37], sigma, size)

        # au17 0.5 scale below lmk 56, 58 / 0.5 scale below lip center
        heatmap = add_noise(heatmap, 9, label, [lmk[39,0], lmk[39,1]+0.5*IOD], [lmk[41,0], lmk[41,1]+0.5*IOD], sigma, size)

        # au23 lmk 51, 57 / lip center
        heatmap = add_noise(heatmap, 10, label, lmk[34], lmk[40], sigma, size)

        # au24 lmk 51, 57 / lip center
        heatmap = add_noise(heatmap, 11, label, lmk[34], lmk[40], sigma, size)
    elif config.DATA.DATASET == 'DISFA':
        # au1 lmk 21, 22
        heatmap = add_noise(heatmap, 0, label, lmk[4], lmk[5], sigma, size)

        # au2 lmk 17, 26
        heatmap = add_noise(heatmap, 1, label, lmk[0], lmk[9], sigma, size)

        # au4 brow center
        heatmap = add_noise(heatmap, 2, label, eyebrow_left, eyebrow_right, sigma, size)

        # au6 1 scale below eye bottom
        heatmap = add_noise(heatmap, 3, label, [eye_left[0], eye_left[1]+IOD], [eye_right[0], eye_right[1]+IOD], sigma, size)

        # au9 0.5 scale below lmk 39, 42 / lmk 39, 42 / lmk 21, 22
        heatmap = add_noise(heatmap, 4, label, lmk[22], lmk[24], sigma, size)

        # au12 lmk 48, 54
        heatmap = add_noise(heatmap, 5, label, lmk[31], lmk[37], sigma, size)

        # au25 lmk 51, 57
        heatmap = add_noise(heatmap, 6, label, lmk[34], lmk[40], sigma, size)

        # au26 0.5 scale below lmk 56, 58 / lmk 56, 58
        heatmap = add_noise(heatmap, 7, label, lmk[39], lmk[41], sigma, size)

    #将 heatmap 矩阵中的所有值限制在 -1.0 到 1.0 之间。
    heatmap = np.clip(heatmap, -1., 1.)

    return heatmap

# 专为 AU（面部动作单元）检测任务设计的自定义数据集实现类
# 多模态数据加载：同时读取图像、面部关键点、双眼距离、AU标记和热图（Heatmap）
# 同步数据预处理：确保图像变换与热图处理严格同步
# 增强随机性控制：通过 RNG 状态恢复机制保持图像与热图变换的一致性
class ImageList(object):

    def __init__(self, crop_size, path, transform=None, target_transform=None, loader=default_loader, config=None):
        # 数据源加载
        image_list = open(path + '_path.txt').readlines()           # 图像路径
        land = np.loadtxt(path + '_land.txt')                       # 面部关键点
        biocular = np.loadtxt(path + '_biocular.txt')               # 双眼距离信息
        au = np.loadtxt(path + '_AUoccur.txt')                      # AU的存在情况
        imgs = make_dataset(image_list, land, biocular, au)
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))
        # 成员变量初始化
        self.imgs = imgs                # 结构化数据元组 (路径、关键点等信息)
        self.transform = transform      # 图像处理流水线
        # 变换隔离性：target_transform 复用图像变换但排除 Normalize，因为热图的数值分布需要独立控制
        self.target_transform = transforms.Compose([
            t for t in self.transform.transforms
            if not isinstance(t, transforms.Normalize)      # 排除标准化层
        ])
        self.loader = loader            
        self.crop_size = crop_size      # 裁剪尺寸
        self.config = config            # 全局配置
        


    def __getitem__(self, index):
        # 数据加载
        path, land, biocular, au = self.imgs[index]
        img = self.loader(path) # 加载原始图像

        # 图像变换应用
        if self.transform is not None:
            img = self.transform(img)

        # heatmap_final = []
        # for idx in range(heatmap.shape[0]):
        #     random.setstate(python_rng_state)
        #     np.random.set_state(np_rng_state)
        #     torch.set_rng_state(torch_rng_state)
        #     heatmap_tmp = heatmap[idx:idx+1].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
        #     heatmap_tmp = (heatmap_tmp * 255).astype('uint8')
        #     heatmap_pil = Image.fromarray(heatmap_tmp, mode='RGB')  # L为灰度模式
        #     if self.transform is not None:
        #         heatmap_tmp = self.target_transform(heatmap_pil)
        #     heatmap_final.append(heatmap_tmp.mean(0, keepdim=True))
        # heatmap = torch.cat(heatmap_final, dim=0)
        # heatmap = F.interpolate(heatmap.unsqueeze(1), size=(112, 112), mode='bilinear', align_corners=False).squeeze(1)

        # random.setstate(python_rng_state)
        # np.random.set_state(np_rng_state)
        # torch.set_rng_state(torch_rng_state)
        # heatmap = self.target_transform(heatmap)


        # # 将每张图片单独保存
        # to_pil = transforms.ToPILImage()
        # image = to_pil(img)  # 假设 img 是形状 (C, H, W)
        # # image.save("output_image.jpg")
        
        # # 保存彩色图像
        # # color_image.save(f"image_transform_{idx}.png")
        # image = image.convert("RGBA")

        # norm_heatmap = (heatmap + 1) / 2.0  # [-1, 1] -> [0, 1]
        # # 确保 Tensor 的数值范围在 [0, 255]，将其转换为 8-bit 图像
        # tensor_heatmap = (norm_heatmap * 255).byte()

        # for i in range(tensor_heatmap.size(0)):
        #     # 将单通道 Tensor 转为 PIL 图像（灰度图像）
        #     gray_image = to_pil(tensor_heatmap[i])  # shape: (H, W)
            
        #     # 转换为 numpy 数组
        #     gray_array = np.array(gray_image)  # uint8, [0, 255]
            
        #     # 将灰度值归一化到 [0, 1] 区间
        #     norm_array = gray_array.astype(np.float32) / 255.0
            
        #     # 使用 matplotlib 的 'jet' 颜色映射
        #     color_mapped = cm.jet(norm_array)  # 返回 (H, W, 4) RGBA
        #     color_mapped = (color_mapped[..., :3] * 255).astype(np.uint8)  # 取RGB，转为uint8
            
        #     # 转回 PIL 图像
        #     color_image = Image.fromarray(color_mapped)

        #     color_image = color_image.convert("RGBA")

        #     combined = Image.blend(image, color_image, alpha=0.5)
        #     combined.save(f"combined_image_{idx}.png")

        # exit()
        return img, land, biocular, au

    # 作用域完善 (__len__)
    def __len__(self):
        return len(self.imgs)   # 实现数据集长度接口