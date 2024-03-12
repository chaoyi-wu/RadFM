# RadFM
The official code for the paper "Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data"

[ArXiv](http://arxiv.org/abs/2308.02463)

[Website](https://chaoyi-wu.github.io/RadFM/)

[Model checkpoint](https://huggingface.co/chaoyi-wu/RadFM)

In this project, we collect a large-scale medical multi-modal dataset, MedMD, with **16M** 2D or **3D** images. We train a new medical multi-modal generative model RadFM on it, enabling both **2D and 3D** scans, multi-image input and visual-language interleaving cases.

<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/GIF.gif"/>

## Latest News：
All Datasets are released! We have updated the links in [our dataset table](#dataset-links). You can find all our text part data in https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv. 

For decompressing the splited compression files in most cases, please check the following code in linux:
```
cat zip.z* > myzip.zip
unzip myzip.zip
```

## Quick Start:

For quick start, you can check the `Quick_demo` path.   
We demonstrate a simple diagnosis case here to show how to inference with our model.   
Feel free to modify it as you want.

- S1. Download [Model checkpoint](https://huggingface.co/chaoyi-wu/RadFM) or form  [baiduyun](https://pan.baidu.com/s/1A-K5nXCbvWAVqvb6dLjYJg?pwd=q1eo) (No need for decompressing).
- S2. Decompress the original zip file, you can get a  `pytorch_model.bin`.
- S3. put `pytorch_model.bin` under path `Quick_demo/`.
- S4. python `test.py` and you can get a conversation as:   

    > Input: <img src="https://github.com/chaoyi-wu/RadFM/blob/main/Quick_demo/view1_frontal.jpg" style="width:15px;"/> Can you identify any visible signs of Cardiomegaly in the image?    
    > Output: yes 

By the way, never try to perform this in cpu and gpu is all you need :）.

## Pre-train:
For re-training a model on our dataset or large-scale testing our pre-train model, you can check ```src```.

Simply, ```train.py``` for training and ```test.py``` for testing.

* Check the [data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) to get how different datasets are processed and download them into `src/Dataset/data_csv` 
* Modify the path as you disire, and check ```src/train.py``` to pre-train or ```src/train.py``` to test.

## Case Study:

Some cases produced by our final model:

<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/result_vqa.jpg"/>
<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/result_report.jpg"/>
<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/result_rationale.jpg"/>

## Dataset-Links:
MedKD Dataset downloading URL:

| Dataset Name | Link | Access |
|--------------|------|--------|
| Rad3D-series | - | Closed |
| MPx-series | - | Closed |
| PMC-Figures| https://pan.baidu.com/s/1Src_rhXsaOFp8zJ_3zMFsQ?pwd=p3ne | Open Access |
| PMC-Inline | https://huggingface.co/datasets/chaoyi-wu/PMC-Inline | Open Access |
| PMC-CaseReport | [Original version](https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport_original), [Filtered version](https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport) | Open Access |
| VinDr-Mammo | https://www.physionet.org/content/vindr-mammo/1.0.0/ | Credentialed Access |
| VinDr-SpineXR | https://www.physionet.org/content/vindr-spinexr/1.0.0/ | Credentialed Access |
| VinDr-PCXR | https://physionet.org/content/vindr-pcxr/1.0.0/ | Credentialed Access |
| PMC-OA | https://huggingface.co/datasets/axiong/pmc_oa_beta | Open Access |
| PMC-VQA | https://huggingface.co/datasets/xmcmic/PMC-VQA | Open Access |
| VQA-RAD | https://osf.io/89kps/| Open Access |
| SLAKE | https://www.med-vqa.com/slake/ | Open Access |
| MIMIC-CXR | https://physionet.org/content/mimic-cxr/2.0.0 | Credentialed Access |
| VinDr-CXR | https://physionet.org/content/vindr-cxr/1.0.0/ | Credentialed Access |
| NIH ChestXray14 | https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345 | Open Access |
| CheXpert | https://aimi.stanford.edu/chexpert-chest-x-rays | Open Access |
| Covid-CXR2 | https://www.kaggle.com/datasets/andyczhao/covidx-cxr2 | Open Access |
| NLM-TB | [Montgomery](https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip), [ChinaSet](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip) | Open Access |
| Object-CXR | https://web.archive.org/web/20201127235812/https://jfhealthcare.github.io/object-CXR/ | Open Access |
| OpenI | https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university | Open Access |
| RSNA| https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018| Open Access |
| SIIM-ACR | https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data| Open Access |

## Acknowledgment:
We sincerely thank all the contributors who uploaded the relevant data in our dataset online. We appreciate their willingness to make these valuable cases publicly available.

## Contact
If you have any questions, please feel free to contact wtzxxxwcy02@sjtu.edu.cn.
