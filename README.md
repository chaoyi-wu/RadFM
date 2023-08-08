# RadFM
The official code for "Towards Generalist Foundation Model for Radiology"

 [ArXiv](http://arxiv.org/abs/2308.02463)

 [Website](https://chaoyi-wu.github.io/RadFM/)

<img src="https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/figures/zero-shot_cases.png"/>

## Quick Start:
Download [Model checkpoint](https://huggingface.co/chaoyi-wu/RadFM) and check `./src/test.py` for how to generate text with our model. 

## Pre-train:
Our pre-train code is given in ```./src/train.py```. 
* Check the [data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) (uploading) to get how different datasets are processed and down load them into `./src/Dataset/data_csv` 
* Modify the path as you disire, and check ```./src/train.py``` to pre-train.

## Case Study:

<img src="https://github.com/chaoyi-wu/RadFM/blob/main/figures/zero-shot_cases.png"/>
<img src="https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/figures/zero-shot_cases.png"/>
<img src="https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/figures/zero-shot_cases.png"/>
## To-do List:
- Polish the code for easier usage.
- Update an easy sample for a quick start.

## Key Links

[Model checkpoint](https://huggingface.co/chaoyi-wu/RadFM)

[data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) (uploading) (For training usage, downlowd into `./src/Dataset/data_csv`)

MedKD Dataset downloading URL:
| Dataset Name | Link | Access |
|--------------|------|--------|
| Rad3D-series | - | Restricted Access |
| MPx-series | - | Restricted Access |
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
