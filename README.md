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

By the way, never try to perform this in cpu and gpus are all you need :）.

## Pre-train:
For re-training a model on our dataset or large-scale testing our pre-train model, you can check ```src```.

Simply, ```train.py``` for training and ```test.py``` for testing.

* Check the [data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) to get how different datasets are processed and download them into `src/Dataset/data_csv` 
* Modify the path as you disire, and check ```src/train.py``` to pre-train or ```src/train.py``` to test.

## A Detailed Code Explanation:
In this part we will introduce the ```src``` directory in detail and the `Quick_demo` is similar.

### Dataset
In the `Dataset` directory, there are two main Python files:

- `multi_dataset.py`
- `multi_dataset_test.py`

These files are nearly identical in structure and functionality. The primary difference lies in their usage: `multi_dataset.py` is used for **training**, while `multi_dataset_test.py` is used for **testing**.

Both files define a key class: `multi_dataset`. This class provides a generative training format that supports multiple datasets. When an instance of this class is called to retrieve a sample, it returns a dictionary with the following structure:
```
{
    'vision_x': vision_x,
    'lang_x':lang_x, 
    'attention_mask': attention_mask, 
    'labels':labels, 
    'loss_reweight': reweight_tensor, 
    'key_words_query': emphasize_words
}
``` 
where, each means:

- **`vision_x`**: A tensor representing input images, shaped as **3 × H × W × D**, where `3` is the number of channels (RGB). If only 2D images are provided, they are repeated along the depth dimension (`D = 4` by default).

- **`lang_x`** and **`attention_mask`**: These represent tokenized text inputs and corresponding attention masks. They may include special image placeholders, which are replaced with image embedding tokens during the model's forward pass.

- **`labels`**: Token IDs corresponding to the output text (e.g., answers). As in standard LLM training, a value of `-100` marks tokens that should be ignored in the loss computation. This supports both reconstruction pretraining and instruction tuning (response-only) loss.

- **`loss_reweight`**: An optional tensor used to emphasize specific medical-related terms (e.g., USMLE keywords). It is applied to the per-token autoregressive loss to compute a weighted final loss.

- **`key_words_query`**: Currently unused. It was part of an earlier experiment involving query-based classification loss. You can safely ignore this field by setting it to an empty list (`[]`).

Then in the sub-directory `dataset`, it contains many detailed dataset-wise classes along with our used prompt for organizing them into generative training style, if you want to see how we prompt different dataset you show check the correponding file carefully. For example the chestxray diagnosis dataset listed in our paper are unfiedly preprocess and prompted in `chestxray.py` with the prompt format listed in `yes_no_prompt.json`.

### My_Trainer and datasampler.py

`My_trainer` is a customized version of the `trainer.py` module from `transformers==4.28.1`. The main motivation for creating this separate trainer file is that the original `Trainer` class does not support passing a custom `data_sampler` during `DataLoader` initialization.

Our goal is to **prevent mixing 2D and 3D data within the same training batch**, which can lead to significant overhead when trying to unify the tensor dimensions. By controlling the sampling strategy, we can avoid unnecessary data expansion and improve training efficiency.

The changes in `My_trainer` are clearly marked with the comment tag `### 吴超逸加 ###`, retained in **Chinese** for easier identification and tracking. These modifications can be integrated into any newer version of the `transformers` library as needed.
The `data_sampler.py` python file contains a new distributed sampling function implemented to ensure proper batch organization. It samples either **2D** or **3D** data exclusively within a single batch. This design avoids the computational cost of dynamically expanding 2D data to match 3D inputs when they are mixed in a batch.

### train.py and test.py

The two python files are easy to understand. `train.py` is used to train the model including pre-training and instruction tuning. `test.py` is used to perform testing on different datset. Please check the [data_csv](https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv) download the used train/test split csv files into `src/Dataset/data_csv` along with the image sources from different dataset official website and ensure the image path witten in the csv files have been changed to your local path, then you can run the `test.py` successfully. Please ensure you have at least one Nvidia A100 (80GB) to surpport the inference, otherwise it will be quite slow that you can never obtain the results. The output csv file will be like that presented in `src/output_csv_example/caption_example.csv` (an output example for chestxray report generation). You can compare your output format with it to check whether your code is right. Notably, in `test.py`. we adopt inference batch size as one by default to avoid some necessary padding. You can change it to a larger size but please ensure your padding tokens~(shoud be left padding) and the attention mask is set correctly according to the classic LLM batch-wise generation guideline. Otherwise the model cannot output correctly due to take the padding token into foward caculation.

### Model

The main python files in the Model path are two, i.e., `RadFM//multimodality_model.py` and `RadFM/my_embedding_layer.py`. In the `multimodality_model.py`, it defines a class `MultiLLaMAForCausalLM`, it is similar to classic `CausalLM` classes. The forward function in this class is response for the LLM-based fusion and decoding process. As shown by the code, it will first call
```
    input_embedding,loss_match= self.embedding_layer(lang_x, vision_x,key_words_query) 
``` 
that the `self.embedding_layer` is defined by `RadFM/my_embedding_layer.py`. The `input_embedding` is the visual-text mixed token embedding sequancing. `loss_mathc` is related to `key_words_query` input and samely, is aborted now, that always equals zero. Then the forward functin will take the `input_embedding` into any LLMs to obtain the final textual generation and calculate the auto-regressive loss based on the input labels and loss_reweight

In `my_embedding_layer.py`, the vision input is first processed using a **3D Vision Transformer (ViT)** and a **Perceiver** model. This results in a set of image tokens with shape **S × 32 × d**, where:

- `S` is the number of images (or scans) in a training sample,
- `32` is a fixed token length per image,
- `d` is the embedding dimension.

These generated image tokens are then appended to the language token embedding layer, resulting in an expanded embedding matrix of shape **(Vocab_size + 32 × S) × d**. This design allows each image token to be referenced using special placeholder token IDs that are defined as vocab_size+n in expanded LLM tokenizer.

By structuring the embedding layer in this way, we canperform the standard token embedding lookup mechanism without requiring explicit loops to insert the token embeddings into textual and reorganize the lenthy embedding output again during the forward pass.

When the batch size `B` is greater than 1, each sample in the batch may have different image tokens. Therefore, during the forward pass, each training sample must be **matched with its own corresponding expanded embedding layer**, i.e., the expanded matrixs is of size **B × (Vocab_size + 32 × S) × d**to correctly replace the image placeholders with the appropriate image encoding features.

## Case Study:

Some cases produced by our final model:

<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/result_vqa.jpg"/>
<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/result_report.jpg"/>
<img src="https://github.com/chaoyi-wu/RadFM/blob/main/Images/result_rationale.jpg"/>

## Dataset-Links:
Datasets downloading URL:

| Dataset Name | Link | Access |
|--------------|------|--------|
| Rad3D-series | - | Please mail the Radiopaedia team to obtain access approvement. Then we can share with you. |
| MPx-series | - | Download from the official websit. |
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

The split of each dataset can be found in https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv you just need to download the image part from each datasets.

## Dataset Codes and Files Linking:
Check the following table to see how to process each dataset and how each file in https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv is linked to each dataset:

| Dataset Name | Process Dataset Code | Related Filename |
|--------------|------|--------|
| Rad3D-series | [jpg2nii Process Code](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/jpg2nii_data_convert.py), [nii2npy Process Code](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/nii2npy_for_radiopaedio.py), [Final Datset to Read npy and Related Texts](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/radiopaedia.py) | radiology_article_npy_train/test.json  |
| MPx-series | [MedPix Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/MedPix_dataset.py) | MedPix_muli_train/test.csv, MedPix_single_train/test.csv|
| PMC-Inline | [Paper-inline Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/paper_inline.py) | paper_train.csv (This dataset is not used for evaluation) |
| PMC-CaseReport | [Case-report Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/case_report.py) | filtered_case_report_train/test.csv |
| VinDr-Mammo | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | mammo_balance_train/test.csv |
| VinDr-SpineXR | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | spinexr_balance_train/test.csv |
| VinDr-PCXR | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | pcxr_balance_train/test.csv |
| PMC-OA | [Pmcoa Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/pmcoa.py) | pmcoa_image_caption_train/test.csv |
| PMC-VQA | [vqa Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/vqa.py) | pmcvaq_train/test.csv|
| VQA-RAD | [vqa Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/vqa.py)| vqarad_train/test.csv |
| SLAKE | [vqa Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/vqa.py) | slakevqa_train/test.csv |
| MIMIC-CXR | [CXR Open Captioning Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py) | mimic_caption_train/test.csv |
| VinDr-CXR | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv|
| NIH ChestXray14 | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| CheXpert | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| Covid-CXR2 | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| NLM-TB | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| Object-CXR | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| OpenI | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| RSNA| [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py)| chestxray_balance_train_new.csv,  chestxray_balance_test.csv |
| SIIM-ACR | [Diagnosis Open Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/chestxray.py), [Diagnosis Close (yes/no) Format Dataset](https://github.com/chaoyi-wu/RadFM/blob/main/src/Dataset/dataset/binary.py) | chestxray_balance_train_new.csv,  chestxray_balance_test.csv|


## Acknowledgment:
We sincerely thank all the contributors who uploaded the relevant data in our dataset online. We appreciate their willingness to make these valuable cases publicly available.

## Contact
If you have any questions, please feel free to contact wtzxxxwcy02@sjtu.edu.cn.
