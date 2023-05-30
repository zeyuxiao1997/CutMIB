CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending
====
Zeyu Xiao, Yutong Liu, Ruisheng Gao, and Zhiwei Xiong. [CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_CutMIB_Boosting_Light_Field_Super-Resolution_via_Multi-View_Image_Blending_CVPR_2023_paper.pdf). In CVPR 2023. <br/>

## Dependencies
- This repository is based on [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) 
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.2.0](https://pytorch.org/): `conda install pytorch=1.2.0 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `pip install numpy`
- opencv: `pip install opencv-python`
- tensorboardX: `pip install tensorboardX`


## Train the model
```
python training_InterNet_cutmib_x4.py
```

## Test the model
```
python inference_InterNet_cutmib_x4.py
```


## Citation
```
@InProceedings{Xiao_2023_cutmib,
    author    = {Xiao, Zeyu and Liu, Yutong and Gao, Ruisheng and Xiong, Zhiwei},
    title     = {CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending},
    booktitle = {CVPR},
    year      = {2023},
}

@InProceedings{Xiao_2023_toward,
    author    = {Xiao, Zeyu and Gao, Ruisheng and Liu, Yutong and Zhang, Yueyi and Xiong, Zhiwei},
    title     = {Toward Real-World Light Field Super-Resolution},
    booktitle = {CVPRW},
    year      = {2023},
}

```

## Contact
Any question regarding this work can be addressed to zeyuxiao1997@163.com or zeyuxiao@mail.ustc.edu.cn.