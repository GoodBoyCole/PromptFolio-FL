# How to install datasets
Our dataset setting is inhereted from [CoOp](https://github.com/KaiyangZhou/CoOp). We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– dtd/
|–– caltech-101/
|–– oxford_pets/
|–– food101/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [DTD](#dtd)


The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is use