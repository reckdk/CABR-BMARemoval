# Self-Supervised Bulk Motion Artifact Removal in Optical Coherence Tomography Angiography

Jiaxiang Ren<sup>1</sup>,
Kicheon Park<sup>2</sup>,
Yingtian Pan<sup>2</sup>,
Haibin Ling<sup>1</sup>

<sup>1</sup>Department of Computer Science, <sup>2</sup>Department of Biomedical Engineering

Stony Brook University

---

This repository is the official Keras implementation of Content-Aware BMA Removal model (CABR). [Paper](https://arxiv.org/abs/2202.10360)

### Environment:

```
tensorflow == 1.14.0
keras == 2.3.1
```

### Inference

Ensure the trained model weights is in `model_weights/bestmodel.hdf5`. Then run

```shell
python -u inference.py
```

DICE score will be printed after inference. 

Find the predicted mask and enhanced image in

```shell
.
└── dataset
    ├── ...
    ├── AwakeOCA_mask_pred_DiceXXXX.tif # Predicted mask
    └── AwakeOCA_enhanced.tif # Enhanced image
```

The rest of dataset used in this work is not available for now due to data policy.

### Citation

```bibtex
@inproceedings{ren2022self,
  title={Self-Supervised Bulk Motion Artifact Removal in Optical Coherence Tomography Angiography},
  author={Ren, Jiaxiang and Park, Kicheon and Pan, Yingtian and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20617--20625},
  year={2022}
}
```

### Contact

jiaxren@cs.stonybrook.edu

### Acknowledgment

This work was supported in part by NSF Grants 1814745 and 2006665, and NIH grants
R01DA029718 and RF1DA048808.
