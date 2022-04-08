

## Multi-Scale Progressive Attention Network for             Video Question Answering  (MSPAN-VideoQA)

![](.\figures\figure2.png)

#### Requirements

Python = 3.6

PyTorch = 1.2

#### Setups

1. Install the python dependency packages:

   ```bash
   pip3 install -r requirements.txt
   ```

   If you can't run `nltk.download('punkt')` to download `punkt`, you could download `punkt.zip` from [here](https://pan.baidu.com/s/1Mq9hoFEy_FdmcQslHeoSEA)(Extraction code: bft3).

2. Download [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA, MSRVTT-QA](https://github.com/xudejing/video-question-answering) datasets and edit absolute paths in `preprocess/question_features.py` , `preprocess/appearance_features.py` and `preprocess/motion_features.py` upon where you locate your data.

#### Preprocessing features

You can download our pre-extracted features from  [here](https://pan.baidu.com/s/1Mq9hoFEy_FdmcQslHeoSEA)(Extraction code: bft3) .

For above three datasets of VideoQA, you can choose 3 options of `--dataset`: `tgif-qa`, `msvd-qa` and `msrvtt-qa`.

For different datasets, you can choose 5 options of `--question_type`: `none`, `action`, `count`, `frameqa` and `transition`.

##### Extracting question features

1. Download [Glove 300D](http://nlp.stanford.edu/data/glove.840B.300d.zip) to `preprocess/pretrained/` and process it into a pickle file:

   ```bash
   python3 preprocess/txt2pickle.py
   ```

2. To extract question features.

   For TGIF-QA dataset:

   ```bash
   python3 preprocess/question_features.py 
           --dataset tgif-qa \
           --question_type action \
           --mode total
   python3 preprocess/question_features.py \
           --dataset tgif-qa \
           --question_type action \
           --mode train
   python3 preprocess/question_features.py \
           --dataset tgif-qa \
           --question_type action \
           --mode test
   ```
   
   For MSVD-QA/MSRVTT-QA dataset:
   
   ```bash
   python3 preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode total
   python3 preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode train
   python3 preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode val
   python3 preprocess/question_features.py \
           --dataset msvd-qa \
           --question_type none \
           --mode test
   ```

##### Extracting visual features

 1. Download pre-trained [3D-ResNet152](https://drive.google.com/file/d/1U7p9iIgkZviuKvpObzN6gx5OiflmAKaT/view?usp=sharing) to `preprocess/pretrained/` .

    You can learn more about this model in the following paper:

    [Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,
    "Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",
    arXiv preprint, arXiv:2004.04968, 2020.](https://arxiv.org/abs/2004.04968)

2. To extract appearance features:

   ```bash
   python3 preprocess/appearance_features.py \
           --gpu_id 0 \
           --dataset tgif-qa \
           --question_type action \
           --feature_type pool5 \
           --num_frames 16
   ```

3. To extract motion features:

   ```bash
   python3 preprocess/motion_features.py \
           --gpu_id 0 \
           --dataset tgif-qa \
           --question_type action \
           --num_frames 16
   ```

For all the above process of extracting features, you can directly run the following command:

```bash
sh pre_sh/action.sh
```

For different datasets and different tasks, there are 6 different options:  `action.sh`,`count.sh`, `frameqa.sh`, `transition.sh`, `msvd.sh` and `msrvtt.sh`.

#### Training

You can choose the suitable `--dataset` and `--question_type` to start training:

```bash
python3 train.py \
        --dataset tgif-qa \
        --question_type action \
        --T 2 \
        --K 3 \
        --num_scale 8 \
        --num_frames 16 \
        --gpu_id 0 \
        --max_epochs 30 \
        --batch_size 64 \
        --dropout 0.1 \
        --model_id 0 \
        --use_test \
        --use_train
```

Or, you can run the following command to start training:

```bash
sh train_sh/action.sh
```

You can see the training commands for all datasets and tasks under the `train_sh` folder.

#### Evaluation

You can download our pre-trained models from  [here](https://pan.baidu.com/s/1Mq9hoFEy_FdmcQslHeoSEA)(Extraction code: bft3) .

To evaluate the trained model, run the following command:

```bash
sh test_sh/action.sh
```

You can see the evaluating commands for all datasets and tasks under the `test_sh` folder.