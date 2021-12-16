# Facial Prior Based First Order Motion Model for Micro-expression Generation
This repository contains the source code for the paper [Facial Prior Based First Order Motion Model for Micro-expression Generation](https://doi.org/10.1145/3474085.3479211) included in [MM '21: Proceedings of the 29th ACM International Conference on Multimedia](https://dl.acm.org/doi/proceedings/10.1145/3474085).

> **FME Challenge 2021 and ACM Multimedia 2021**  
>
> Official Site: https://megc2021.github.io/index.html

## 0. Table of Contents

* [1. Authors & Maintainers](#1-authors---maintainers)

* [2. Change Log](#2-change-log)

* [3. Results in GIF](#3-results-in-gif)

* [4. Run the Code](#4-run-the-code)

* [5. Additional Resources](#5-additional-resources)

  * [5.1 Training Datasets](#51-training-datasets)
  * [5.2 Source Videos & Target Faces for Evaluation in MEGC2021](#52-source-videos---target-faces-for-evaluation-in-megc2021)
  * [5.3 Pre-trained Checkpoint](#53-pre-trained-checkpoint)

* [6. License](#6-license)

* [7. Citation](#7-citation)

  

## 1. Authors & Maintainers

- [Yi Zhang|@zylye123](https://github.com/zylye123)
- [Youjun Zhao|@zhaoyjoy](https://github.com/zhaoyjoy)
- [Yuhang Wen|@Necolizer](https://github.com/Necolizer)
- [Zixuan Tang|@sysu19351118](https://github.com/sysu19351118)
- [Xinhua Xu|@sysu19351158](https://github.com/sysu19351158)



## 2. Change Log

- [2021/12/16] Readme & Filename & Preprocess Utils Updated. Additional resources have been provided in **Section 5**.
- [2021/10/20] Our paper has been published online at https://doi.org/10.1145/3474085.3479211.
- [2021/10/13] Code Updated.
- [2021/07/12] Qualitative results are presented in GIF format.
- [2021/07/10] Create repository, add basic information, and upload code.



## 3. Results in GIF

| No.  |                            Source                            |                             FOMM                             |                             MRAA                             |                            OurSA                             |                            OurSM                             |                            OurCA                             |                            OurMX                             |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |    <img src="./sup-mat/022_3_3.gif" style="zoom: 33%;" />    | <img src="./sup-mat/022_3_3_FOMM.gif" style="zoom: 33%;" />  | <img src="./sup-mat/022_3_3_MRAA.gif" style="zoom: 33%;" />  | <img src="./sup-mat/Positive_022_3_3_SAMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_022_3_3_SMIC.gif" style="zoom: 33%;" /> | <img src="./sup-mat/022_3_3_CASME2.gif" style="zoom: 33%;" /> |  <img src="./sup-mat/022_3_3_mix.gif" style="zoom: 33%;" />  |
|  2   |   <img src="./sup-mat/s3_po_05.gif" style="zoom: 33%;" />    | <img src="./sup-mat/s3_po_05_FOMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/s3_po_05_MRAA.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_s3_po_05_SAMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_s3_po_05_SMIC.gif" style="zoom: 33%;" /> | <img src="./sup-mat/s3_po_05_CASME2.gif" style="zoom: 33%;" /> | <img src="./sup-mat/s3_po_05_MIX.gif" style="zoom: 33%;" />  |
|  3   | <img src="./sup-mat/sub19_EP01_01f.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_FOMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_MRAA.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_EP01_01f_SAMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_EP01_01f_SMIC.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_CASME2.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_MIX.gif" style="zoom: 33%;" /> |

> **NOTICE**: For visual presentation, all the GIFs have been **slowed down**.
>
> `Source` == Source videos, also called driving videos
>
> `FOMM` == [First Order Motion Model for Image Animation](https://github.com/AliaksandrSiarohin/first-order-model)
>
> `MRAA` == [Motion Representations for Articulated Animation](https://github.com/snap-research/articulated-animation)
>
> `OurSA` == Our method, training on SAMM
>
> `OurSM` == Our method, training on SMIC-HS
>
> `OurCA` == Our method, training on CASMEⅡ
>
> `OurMIX` == Our method, training on SAMM + SMIC-HS + CASMEⅡ



## 4. Run the Code

1. Installation. We support `python3`. To install the dependencies run:

   ```she
   pip install -r requirements.txt
   ```

   > P.S. The Original FOMM `requirments.txt`  says that `torch==1.0.0` , but it seems that `torch` whose version is higher than 1.0.0 also works, provided that all the dependency conflicts are solved.

2. Prepare your own dataset. Divide it into `./data/your_dataset/train` and `./data/your_dataset/test`.

   - For our task, [CASMEⅡ](http://fu.psych.ac.cn/CASME/casme2.php), [SAMM](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php) and [SMIC-HS](https://www.oulu.fi/cmvs/node/41319) are used for training. See **Section 5.1**.
   - In MEGC 2021 FME Generation Task,  particular *Source Videos & Target Faces* are provided for test/evaluation. See **Section 5.2**.

3. Create new  `yaml` format files, or modify `your_dataset_train.yaml` & `your_dataset_test.yaml` in `./config`. Set your expected evaluation pair configuration by modify `./data/your_dataset_test.csv`, which indicates which videos serve as source videos and which images serve as target faces.

   - If you would like to take the same *Training & Testing Data* as ours, there's no need for modifications to these files.

4. Preprocess your dataset. 

   The input frame shape is (256, 256, 3) (though you could change it by altering `your_dataset_train.yaml` & `your_dataset_test.yaml` in `./config`). We provide preprocess utils in  `./preprocess.py` that might be helpful when preprocessing your dataset. The following steps illustrate how we did it for our training & testing data.

   1. Use `dlib.get_frontal_face_detector()` to detect the face in the onset frame and get its bounding box. This would treat as the face bounding box of the whole micro-expression video, as in FME videos, head movements are really subtle.  The bounding box of the onset frame is usually large enough.
   2. Crop sequence of video frames using the 1st bounding box and resize them to $256\times 256$.
   3. (optional) For each video frame, convert RGB to GARY, and then in channel dimension merge itself 3 times to get a 3-channel grayscale. The reason why we did it is that, video frames in SAMM dataset are in this form while the other two datasets are not. To train on SAMM + SMIC-HS + CASMEⅡ and ensure the consistency of input data, we did it for SMIC-HS  and CASMEⅡ, though the original data are colored images. **It is worth noting that training on grayscale version of dataset would results in grayscale generated videos. This step is optional if you want colored generation results.**
   4. Store frame sequence of each FME video in each individual folder.

5. If you take the same *Training & Testing Data* as ours, your final directory tree should look like this:

   ```
   Facial-Prior-Based-FOMM
   ├── data
   |   ├── your_dataset_test.csv
   |   └── your_dataset
   |    	├── train
   |       |   ├── 006_1_2
   |       |   |	├── 006_05562.png
   |       |   |	├── 006_05563.png
   |       |   |   ├── ...
   |       |   ├── 006_1_3
   |       |   ├── ... 
   |       └── test
   |           ├── normalized_westernMale
   |           |   └── normalized_westernMale_gray.png
   |           ├── normalized_asianFemale
   |           ├── Surprise_007_7_1
   |           |   ├── 007_7030.png
   |           |   ├── ...
   |           ├── Positive_022_3_3
   |           ├── Negative_018_3_1
   |           ├── Surprise_EP01_13
   |           ├── Positive_EP01_01f
   |           ├── Negative_EP19_06f
   |           ├── Surprise_s20_sur_01
   |           ├── Positive_s3_po_05
   |           └── Negative_s11_ne_02
   ├── config
   |   ├── your_dataset_train.yaml
   |   └── your_dataset_test.yaml
   ├── ...
   ```

   In `./data/your_dataset/test`, subfolders `normalized_westernMale` & `normalized_asianFemale` serve as *target faces*. Each of them only contains one face image. The other subfolders serve as *source videos*, holding sequences of video frames.

6. Run `kpmaker.py`.

   ```shell
   python kpmaker.py
   ```

   Key points of each video in your dataset would be generated in`./keypoint_folder`.

7. Train

   ```shell
   python run.py --config config/your_dataset_train.yaml
   ```

   Log and checkpoints would be saved in `./log`.

8. Test/Evaluation

   ```shell
   python run.py --config config/your_dataset_test.yaml --mode animate --checkpoint path-to-checkpoint
   ```

   Generated videos would be saved in `path-to-checkpoint/animation`.

> **NOTICE**: [Github repository of FOMM](https://github.com/AliaksandrSiarohin/first-order-model) provides details about how to run the code using multi-GPU and many other detailed tricks. If you encounter problems when running this code, please refer to that repository. And if your problem still exists, please let us know by creating a new issue.



## 5. Additional Resources

### 5.1 Training Datasets

| Datasets                 | Website Link                                       | Online Paper                                 |
| ------------------------ | -------------------------------------------------- | -------------------------------------------- |
| CASMEⅡ                   | http://fu.psych.ac.cn/CASME/casme2.php             | https://pubmed.ncbi.nlm.nih.gov/24475068/    |
| SAMM                     | http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php | https://ieeexplore.ieee.org/document/7492264 |
| SMIC (including SMIC-HS) | https://www.oulu.fi/cmvs/node/41319                | https://ieeexplore.ieee.org/document/6553717 |

### 5.2 Source Videos & Target Faces for Evaluation in MEGC2021

Source videos and target faces for evaluation (pls put them in `your_dataset/test`) can be found at https://megc2021.github.io/images/MEGC2021_generation.zip , which is also provided at https://megc2021.github.io/agenda.html , in section **Description of Challenge tasks**.

### 5.3 Pre-trained Checkpoint

Checkpoint that achieves the final results in our paper can be found under following link: [Google Drive](https://drive.google.com/file/d/15U531P0h2ujql_ag2-mht2tjUlJFYZ_H/view?usp=sharing).

It was trained on the composite dataset of CASME2, SAMM and SMIC-HS.



## 6. License

[MIT](https://github.com/Necolizer/Facial-Prior-Based-FOMM/blob/main/LICENSE)



## 7. Citation

If you find this work or code helpful in your research, please consider citing:

```
@inproceedings{10.1145/3474085.3479211,
author = {Zhang, Yi and Zhao, Youjun and Wen, Yuhang and Tang, Zixuan and Xu, Xinhua and Liu, Mengyuan},
title = {Facial Prior Based First Order Motion Model for Micro-Expression Generation},
year = {2021},
isbn = {9781450386517},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3474085.3479211},
doi = {10.1145/3474085.3479211},
abstract = {Spotting facial micro-expression from videos finds various potential applications
in fields including clinical diagnosis and interrogation, meanwhile this task is still
difficult due to the limited scale of training data. To solve this problem, this paper
tries to formulate a new task called micro-expression generation and then presents
a strong baseline which combines the first order motion model with facial prior knowledge.
Given a target face, we intend to drive the face to generate micro-expression videos
according to the motion patterns of source videos. Specifically, our new model involves
three modules. First, we extract facial prior features from a region focusing module.
Second, we estimate facial motion using key points and local affine transformations
with a motion prediction module. Third, expression generation module is used to drive
the target face to generate videos. We train our model on public CASME II, SAMM and
SMIC datasets and then use the model to generate new micro-expression videos for evaluation.
Our model achieves the first place in the Facial Micro-Expression Challenge 2021 (MEGC2021),
where our superior performance is verified by three experts with Facial Action Coding
System certification. Source code is provided in https://github.com/Necolizer/Facial-Prior-Based-FOMM.},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {4755–4759},
numpages = {5},
keywords = {facial micro-expression, facial landmark, deep learning, micro-expression generation, generative adversarial network},
location = {Virtual Event, China},
series = {MM '21}
}
```
