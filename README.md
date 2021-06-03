# S2R-DepthNet: Learning a Generalizable Depth-specific Structural Representation

This is the official implementation of the paper [***S2R-DepthNet: Learning a Generalizable Depth-specific Structural Representation***](https://arxiv.org/abs/2104.00877), ***CVPR 2021 (Oral), Xiaotian Chen, Yuwang Wang, Xuejin Chen, and Wenjun Zeng.***


## Citation

```
@inproceedings{Chen2021S2R-DepthNet,
             title = {S2R-DepthNet: Learning a Generalizable Depth-specific Structural Representation},
             author = {Chen, Xiaotian and Wang , Yuwang and Chen, Xuejin and Zeng, Wenjun},
	     conference={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
             year = {2021}   
}
```

## Introduction
Human can infer the 3D geometry of a scene from a sketch instead of a realistic image, which indicates that the spatial structure plays a fundamental role in understanding the depth of scenes. We are the first to explore the learning of a depth-specific structural representation, which captures the essential feature for depth estimation and ignores irrelevant style information. Our S2R-DepthNet (Synthetic to Real DepthNet) can be well generalized to unseen real-world data directly even though it is only trained on synthetic data. S2R-DepthNet consists of: a) a Structure Extraction (STE) module which extracts a domaininvariant structural representation from an image by disentangling the image into domain-invariant structure and domain-specific style components, b) a Depth-specific Attention (DSA) module, which learns task-specific knowledge to suppress depth-irrelevant structures for better depth estimation and generalization, and c) a depth prediction module (DP) to predict depth from the depth-specific representation. Without access of any real-world images, our method even outperforms the state-of-the-art unsupervised domain adaptation methods which use real-world images of the target domain for training. In addition, when using a small amount of labeled real-world data, we achieve the state-ofthe-art performance under the semi-supervised setting.
![figure](./img/overview.png)

### Train
As an example, use the following command to train S2RDepthNet on vKITTI.<br>
#### Train TrainStructDecoder

	python train.py --syn_dataset VKITTI
		        --syn_root "the path of vKITTI dataset"
		        --syn_train_datafile datasets/vkitti/train.txt
		        --batchSize 32
		        --loadSize 192 640
		        --Shared_Struct_Encoder_path "the path of pretrained Struct encoder(.pth)"
		        --trian_stage TrainStructDecoder
						       
						       
						       
						                      
