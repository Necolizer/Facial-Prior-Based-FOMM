# Facial Prior Based First Order Motion Model for Micro-expression Generation
This repository contains the source code for the paper [Facial Prior Based First Order Motion Model for Micro-expression Generation](https://doi.org/10.1145/3474085.3479211) included in [MM '21: Proceedings of the 29th ACM International Conference on Multimedia](https://dl.acm.org/doi/proceedings/10.1145/3474085).

> **FME Challenge 2021 and ACM Multimedia 2021**  
>
> Official Site: https://megc2021.github.io/index.html

## 0. Table of Contents

* [0. Table of Contents](#0-table-of-contents)

* [1. Authors & Maintainers](#1-authors---maintainers)

* [2. Change Log](#2-change-log)

* [3. Results in GIF](#3-results-in-gif)

* [4. Run the Code](#4-run-the-code)

* [5. License](#5-license)

* [6. Citation](#6-citation)

  

## 1. Authors & Maintainers

- [Yi Zhang|@zylye123](https://github.com/zylye123)
- [Youjun Zhao|@zhaoyjoy](https://github.com/zhaoyjoy)
- [Yuhang Wen|@Necolizer](https://github.com/Necolizer)
- [Zixuan Tang|@sysu19351118](https://github.com/sysu19351118)
- [Xinhua Xu|@sysu19351158](https://github.com/sysu19351158)

## 2. Change Log

- [2021/10/20] Our paper has been published online at https://doi.org/10.1145/3474085.3479211.
- [2021/10/13] Code Updated.
- [2021/07/12] Qualitative results are presented in GIF format.
- [2021/07/10] Create repository, add basic information, and upload code.

## 3. Results in GIF

> **NOTICE**: For visual presentation, all the GIFs have been **slowed down**.

| No.  |                            Source                            |                             FOMM                             |                             MRAA                             |                            OurSA                             |                            OurSM                             |                            OurCA                             |                            OurMX                             |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |    <img src="./sup-mat/022_3_3.gif" style="zoom: 33%;" />    | <img src="./sup-mat/022_3_3_FOMM.gif" style="zoom: 33%;" />  | <img src="./sup-mat/022_3_3_MRAA.gif" style="zoom: 33%;" />  | <img src="./sup-mat/Positive_022_3_3_SAMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_022_3_3_SMIC.gif" style="zoom: 33%;" /> | <img src="./sup-mat/022_3_3_CASME2.gif" style="zoom: 33%;" /> |  <img src="./sup-mat/022_3_3_mix.gif" style="zoom: 33%;" />  |
|  2   |   <img src="./sup-mat/s3_po_05.gif" style="zoom: 33%;" />    | <img src="./sup-mat/s3_po_05_FOMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/s3_po_05_MRAA.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_s3_po_05_SAMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_s3_po_05_SMIC.gif" style="zoom: 33%;" /> | <img src="./sup-mat/s3_po_05_CASME2.gif" style="zoom: 33%;" /> | <img src="./sup-mat/s3_po_05_MIX.gif" style="zoom: 33%;" />  |
|  3   | <img src="./sup-mat/sub19_EP01_01f.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_FOMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_MRAA.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_EP01_01f_SAMM.gif" style="zoom: 33%;" /> | <img src="./sup-mat/Positive_EP01_01f_SMIC.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_CASME2.gif" style="zoom: 33%;" /> | <img src="./sup-mat/sub19_EP01_01f_MIX.gif" style="zoom: 33%;" /> |

`Source` == Source videos, also called driving videos

`OurSA` == Our method, training on SAMM

`OurSM` == Our method, training on SMIC-HS

`OurCA` == Our method, training on CASMEⅡ

`OurMIX` == Our method, training on SAMM + SMIC-HS + CASMEⅡ

## 4. Run the Code

1. Prepare your dataset. CASME2, SAMM, SMIC-HS are recommended.

   Divide into `your_dataset/train` and `your_dataset/test`.

   Create or modify `yaml` format file `your_dataset_train.yaml` in `./config`.

2. Run `kpmaker.py`.

   ```shell
   python kpmaker.py
   ```

   Key points of each video in your dataset would be generated in`./keypoint_folder`.

3. Train

   ```shell
   python run.py --config config/your_dataset_train.yaml
   ```

   Log and parameters would be saved in `./log`.

4. Test

   ```shell
   python run.py --config config/your_dataset_test.yaml --mode animate --checkpoint path-to-checkpoint
   ```

   Generated videos would be saved in `path-to-checkpoint/animation`.

## 5. License

[MIT](https://github.com/Necolizer/Facial-Prior-Based-FOMM/blob/main/LICENSE)

## 6. Citation

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
