# RadFM
The official code for "Towards Generalist Foundation Model for Radiology"

## Key Links

[Model checkpoint](https://huggingface.co/chaoyi-wu/RadFM)

[data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) (For training usage, downlowd into `./src/Dataset/data_csv`)

MedKD Dataset dowloading URL:
| Dataset Name | Link | Access |
|--------------|------|--------|
| Rad3D-series | - | Restricted Access |
| MPx-series | - | Restricted Access |
| PMC-Inline | [Link](https://huggingface.co/datasets/chaoyi-wu/PMC-Inline) | Open Access |
| PMC-CaseReport | [Link 1](https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport_original) [Link 2](https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport) | Open Access |
| VinDr-Mammo~\cite{vindrmammo} | [Link](https://www.physionet.org/content/vindr-mammo/1.0.0/) | Credentialed Access |
| VinDr-SpineXR~\cite{vindrspine} | [Link](https://www.physionet.org/content/vindr-spinexr/1.0.0/) | Credentialed Access |
| VinDr-PCXR~\cite{vindrpcxr} | [Link](https://physionet.org/content/vindr-pcxr/1.0.0/) | Credentialed Access |
| PMC-OA~\cite{lin2023pmc} | [Link](https://huggingface.co/datasets/axiong/pmc_oa_beta) | Open Access |
| PMC-VQA~\cite{Zhang2023PMCVQAVI} | [Link](https://huggingface.co/datasets/xmcmic/PMC-VQA) | Open Access |
| VQA-RAD~\cite{lau2018dataset} | [Link](https://osf.io/89kps/) | Open Access |
| SLAKE~\cite{liu2021slake} | [Link](https://www.med-vqa.com/slake/) | Open Access |
| MIMIC-CXR~\cite{johnson2019mimic} | [Link](https://physionet.org/content/mimic-cxr/2.0.0) | Credentialed Access |
| VinDr-CXR~\cite{nguyen2022vindr} | [Link](https://physionet.org/content/vindr-cxr/1.0.0/) | Credentialed Access |
| NIH ChestXray14~\cite{wang2017chestx} | [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) | Open Access |
| CheXpert~\cite{irvin2019chexpert} | [Link](https://aimi.stanford.edu/chexpert-chest-x-rays) | Open Access |
| Covid-CXR2~\cite{pavlova2021covid} | [Link](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) | Open Access |
| NLM-TB~\cite{jaeger2014two} | [Link 1](https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip) [Link 2](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip) | Open Access |
| Object-CXR~\cite{objectcxr} | [Link](https://web.archive.org/web/20201127235812/https://jfhealthcare.github.io/object-CXR/) | Open Access |
| OpenI~\cite{demner2016preparing} | [Link](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) | Open Access |
| RSNA~\cite{shih2019augmenting}| [Link](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)| Open Access |
| SIIM-ACR~\cite{filice2020crowdsourcing} | [Link](https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data)| Open Access |

## Quick Start:
Downlad [Model checkpoint](https://huggingface.co/chaoyi-wu/RadFM) and check `./src/test.py` for how to generate text with our model. 

BTW, make sure you have a large GPU. :)

## Pre-train:
Our pre-train code is given in ```./src/train.py```. 
* Check the [data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) to get how different datasets are processed and down load them into `./src/Dataset/data_csv` 
* Modify the path as you disire, and check ```./src/train.py``` to pre-train.

## Contact
If you have any question, please feel free to contact wtzxxxwcy02@sjtu.edu.cn.
