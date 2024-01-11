# FQCDTrans
Architecture of CDTrans with FQ-ViT modules &amp; Apply to PSAQ-ViT




<h1 align="center">FQCDTrans</h1>

## Summary

This repository provides an architecture of [CDTrans](https://github.com/CDTrans/CDTrans), which is the model for domain adaptation, incorporated with a full integer PTQ model [FQ-ViT](https://github.com/megvii-research/FQ-ViT) for providing code for quantizing the domain-adapted model. Not only the architecture, but also provides a code for generating a calibration set for PTQ via [PSAQ-ViT](https://github.com/zkkli/PSAQ-ViT). 

## Instructions
### Code Structure   
* ```utils``` : utils for generating images and domain adaptation
* ```arch_fq_cd.py``` : architecture of FQCDTrans
* ```fqvit_quant.py``` : quantization module from FQ-ViT
* ```fqvit_utils.py``` util functions for FQ-ViT
* ```generate_data.py``` functions for generating the calibration images
* ```evaluation.py``` evaluation code for CDTrans
* ```test_cdtrans_quant.py``` main function with a pipeline of loading pre-trained checkpoint of CDTrans, generating calibration set via PSAQ-ViT, operating PTQ using FQ-ViT modules

### Installation and Requirements
* Clone the repository
```
git clone https://github.com/sehyunpark99/FQCDTrans.git
cd FQCDTrans
```
* Install dependencies
```
conda env create -n fqcdtrans
conda activate fqcdtrans
pip install -r requirements.txt
```
* Prepare datasets: [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)

### Run File
#### Inference
**1) Using run.sh**   
- Fill in the experimental settings in run.sh
```
    bash run.sh
```

**2) Using train.py**
```
    python test_cdtrans_quant.py --model_name 'ar2cl' --config_file '/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/configs/ar2cl.yml' 
```


