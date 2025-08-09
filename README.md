# GMBM
Multi-Attribute Bias Mitigation via Representation Learning
## INSTRUCTIONS TO RUN CODE ON COCO
Train a bias-capturing classifier using `coco_train_bcc.py`:  
- Update the dataset paths in the train and test loaders  
- Run: `python3 coco_train_bcc.py`

Train a model following the methodology described in the paper using `coco_train_bgs.py`:  
- Update the dataset paths in the train and test loaders  
- Set the correct path to the BCC file to be used during training  
- Run: `python3 coco_train_bgs.py`  
- The trained model will be saved as `final_model.pth`

Evaluate the trained model for accuracy and SBA metrics using `evaluate_coco.py`:  
- Update the dataset path in the test loader  
- Set the path to the trained model (`final_model.pth`)  
- Run: `python3 evaluate_coco.py`

Evaluate the trained model for MABA and its variants using `evaluate_coco_maba2.py`:  
- Update the dataset paths in both train and test loaders  
- Set the path to the trained model (`final_model.pth`)  
- Run: `python3 evaluate_coco_maba2.py`


