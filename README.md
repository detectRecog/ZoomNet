# ZoomNet(AAAI2020 oral)
This is the repository for paper "ZoomNet: Part-Aware Adaptive Zooming Neural Network for 3D Object Detection".

The pixel-wise annotations on the KITTI trainval set is available via:
 - [Google Drive trainset](https://drive.google.com/open?id=1vqSrOiwojYiGwuw5wympxw_PC448g6Zf) , [Google Drive valset](https://drive.google.com/open?id=1D79oIocq2hYTtJn3xzH1Iz-yW32CyIZL) 
 
 - [BaiduYun Drive trainset (code: ljhj)](https://pan.baidu.com/s/1Bp0ulfCm6RYvQ770U1V6Eg), [BaiduYun Drive valset (code: kwxf)](https://pan.baidu.com/s/1GMbhj9oHtgqTqij36Acksg)


Sample code for processing the provided annotations.
```python
import pickle
import numpy as np

def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

info = load_pickle(‘000002.pkl) # info.keys(): dict_keys(['data_idx', 'objects', 'calib', 'imw', 'imh', 'instL', 'instR’]). ‘InstL’ contains the pixel-wise inst_id(1-channel), depth(1-channel), part location (3-channel).
# sample code for pixel-wise depth annotation
pkl_objects = info['objects']
calib = info['calib']
f = calib['P'][0, 0]
bl = (calib['P'][0, 3] - calib['P3'][0, 3]) / f
f_bl = f * bl
inst_map_left_ = np.concatenate([np.expand_dims(el.toarray(), -1) for el in info['instL']], axis=-1)
# convert depth to disp
dispMapL = f_bl / inst_map_left_[:, :, 1].copy()
dispMapL[np.isinf(dispMapL)] = 0
print(dispMap.shape)
```

The code for generating pixel-wise annotations and ZoomNet (pytorch) needs to be organised before release. A version on paddle-paddle is also expected to be released. However, I’m currently working on a workshop on CVPR and thus delayed the release of code. I'm sorry about that.

If you are benefited from this paper, please cite our paper as follows:

```
@inproceedings{xu2020zoomnet,
  title={ZoomNet: Part-Aware Adaptive Zooming Neural Network for 3D Object Detection},
  author={Xu, Zhenbo and Zhang, Wei and Ye, Xiaoqing and Tan, Xiao and Yang, Wei and Wen, Shilei and Ding, Errui and Meng, Ajin and Huang, Liusheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={2},
  pages={7},
  year={2020}
}
```


