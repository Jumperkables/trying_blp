# Trying Bilinear Pooling in Video-QA:

The official github repository for [Trying Bilinear Pooling in Video-QA](localhost:8097). This repository uses code adapted from the [TVQA](https://github.com/jayleicn/TVQA.git) and [Heterogeneous Memory Enhanced VQA](https://github.com/fanchenyou/HME-VideoQA) repositories.

## Usage:

0. `git clone https://github.com/Jumperkables/kable_management/tree/master/trying_blp.git`

Our paper surveys 4 datasets across 2 models. In total you'll need to prepare the following 5 main subdirectories and manage 4 datasets. The TVQA model scripts all use one python 3 virtual enviroment, and all the others (including HME-TVQA) share a python 2 virtual enviroment.

### TVQA:

We have merged our TVQA code for this project with another of our projects, [On Modality Bias in the TVQA Dataset](https://github.com/Jumperkables/kable_management/tree/master/projects/tvqa_modality_bias). Follow the setup instructions for this repository, including data collection, and leave it in this position under the name 'tvqa'.<br>
<strong>To reproduce the experiments in this paper you will not need to extract regional features. Feel free to skip that rather complicated step.</strong>

### HME-TVQA:

Our experiments include using TVQA on the HME model. You will need to extract motion vectors from the TVQA raw frames. HME-TVQA will share a dataset directory with TVQA. Our C3D feature extraction code is adapted from [this respoitory](https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor).<br>

0. Apply for access to the raw TVQA frames as detailed in the TVQA subsection.

1. Create another virtual enviroment (Python 2, not my choice im afraid): `pip install hme_tvqa/pytorch_c3d_extraction/requirements.txt`

2. Collect the c3d.pickle pretrained model from the above repository and put it in the `hme_tvqa/pytorch_c3d_extraction` directory.

3. Extraction is done by `pytorch_c3d_extraction/feature_extractor_frm.py`, and can be ran by `c3d.sh`. Set appropriate values for `--OUTPUT_DIR, --VIDEO_DIR, --OUTPUT_NAME`. 

4. The extracted h5 file is in a slightly incorrect format, use `pytorch_c3d_extraction/fix_h5.py` (edit `old` and `new`) to fix h5 file.

5. In your TVQA data directory, add a new subdirectory 'motion_features', and place `fixed_tvqa_c3d_fc6_features.h5` (or whatever you named the fixed h5 file) into it.

6. Since HME is implemented in Python 2, convert `det_visual_concepts_hq.pickle`, `word2idx.pickle`, `idx2word.pickle` and `vocab_embedding.pickle` to a Python 2 compatible format. We have implemented a tool in `tvqa/tools/pickle3topickle2.py` that should do this for you.

### EgoVQA:

0. Collect EgoVQA from [here.](https://github.com/fanchenyou/EgoVQA/blob/master/README.md)

### TGif-QA and MSVD-QA:

0. Collect these two datasets from [this repository.](https://github.com/fanchenyou/HME-VideoQA)


### Models:

This directory contains the adapted models used for all HME based experiments.

### Scripts:

As previously mentioned, the TVQA model scripts use a python 3 virtual enviroment, and the other scripts (including HME-TVQA) all share a different python 2 virtual enviroment. Example scripts to run the experiments in our paper can be found with the in `scripts` with the same hyperparameters.

## Example Dataset Structure:
When collected, your datasets should look something like this.

## Citation:

@inproceedings{tryingblp,
  title={Trying Bilinear Pooling in Video-QA},
  author={Winterbottom, T. and Al-Moubayed, N and Xiao, S. and McLean, A.},
  booktitle={Someone},
  year={2020}
}

## Help:

Feel free to contact us @ thomas.i.winterbottom@durham.ac.uk
