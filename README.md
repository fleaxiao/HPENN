# Hierarchical Physics-Embedding Neural Network Framework for 3D Magnetic Modeling of Medium Frequency Transformers

Hierarchical Physics-Embedding Neural Network (HPENN) Framework is a typically designed tool for the 3D Magnetic Modeling of Medium Frequency Transformers (MFTs).

This respository includs all the necessary elements for training, testing, and using HPINN framework.

## Setup
- Open a terminal and change directory to this program folder:
```bash
cd [path-to-folder]
```
- Clone the temperatur conversion repository:
```bash
git clone https://github.com/fleaxiao/HPENN.git
```
- Implementation order:
```
1. data_generation_2D -> train_2D_loss -> test_2D_loss
2. model_combine
3. data_generation_3D -> train_3D_loss -> test_3D_loss
4. Use_HPENN
```
## Citation
If you use this code or data in your research, please cite our work as follows:
```
Xiao Yang, Liangcai Shu, Dongsheng Yang. Hierarchical Physics-Informed Neural Network Framework for 3D Magnetic Modeling of Medium Frequency Transformers. TechRxiv. April 29, 2024.
```
