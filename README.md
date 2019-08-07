# TDD-Net defect_classifier
Convolutional Neural Network for topological defects classification
## Getting started
Installing required dependencies
```
pip3 install -r requirements.txt
```
- Move all images to ```data/``` Note that all images should be named as XXXXXX.jpg where XXXXXX is a 6-digit index starting with 1

- Obtaining labeled data in Dataframe format stored in a csv file. This dataframe should have following columns:
  ```image_index | class | x | y ```
