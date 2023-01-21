## Descriptions for built-in datasets

### VE-LOL
[download link](https://flyywh.github.io/IJCV2021LowLight_VELOL/)  

File Structure of VE-LOL:
```
VE-LOL
└───VE-LOL-L-Syn
|   └───VE-LOL-L-Syn-Low_test
|   └───VE-LOL-L-Syn-Low_train
|   └───VE-LOL-L-Syn-Normal_test
|   └───VE-LOL-L-Syn-Normal_train
└───VE-LOL-L-Cap-Full
|   └───VE-LOL-L-Cap-Low_test
|   └───VE-LOL-L-Cap-Low_train
|   └───VE-LOL-L-Cap-Normal_test
|   └───VE-LOL-L-Cap-Normal_train
└───VE-LOL-H
    └───train_images
    └───train_labels
```
Splits of VE-LOL:
- "ve_lol_syn_train": training set of the synthesized VE-LOL dataset, which contains 900 image pairs.
- "ve_lol_syn_test": testing set of the synthesized VE-LOL datasets, which contains 100 image pairs.
- "ve_lol_real_train": training set of the real captured VE-LOL dataset, which contains 400 image pairs.
- "ve_lol_real_test": testing set of the real captured VE-LOL dataset, which contains 100 image pairs.
- "ve_lol_all": all splits of the synthesized and real captured VE-LOL datasets, which contains 1500 image pairs.


### SDSD
SDSD is a video low-light dataset, the author provides two format of data: (1) numpy array, all video frames are saved as `.npy` file and the resolution is (512, 960) for fast training; (2) image file, all video frames are saved as `.png` file and the resolution is (1080, 1920).  
You can either train with numpy files or original images without having to prepare all these two formats data before training. No matter what data format you choose, you need to prepare the corresponding dataset files in the following file structure:
```
SDSD/
  # numpy array file structure
  indoor_np/
    GT/
      pair.../
    input/
      pair.../
  outdoor_np/
    GT/
      pair.../
    input/
      pair.../

  # original image file structure
  # Not Implemented Now
```
NOTE: Currently only numpy array format is supported.

