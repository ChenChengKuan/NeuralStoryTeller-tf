# Story Decoder

This code is adapted from [tensorflow im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt). Thanks for google team's great work. The current pretrained model can be download from [here](https://drive.google.com/file/d/1EddIyRZ1D6eamxfxZ2OIwrAhiJpiC16Q/view?usp=sharing). Following steps will guide you how to train the decoder from scratch

### Step 1: Build training data
First run the command below to build up the training data:
```
python build_data.py --stv_vocab "SKIPTHOUGHT_PRETRAINED_PATH/vocab.txt"
                     --stv_embedding "SKIPTHOUGHT_PRETRAINED_PATH/embeddings.npy"
                     --stv_model "SKIPTHOUGHT_PRETRAINED_PATH/model-xxxxxx.ckpt"
                     --book_data_dir "BOOKCORPUS_PATH"
                     --output_dir "TRAIN_DATA_PTH"
                     --book_data_cat "book category name"
```
To speed up the training, current work will first encode all passages into skipthought embedding and store it as `TRAIN_DATA_PATH/stv_text_code.tfrecord`

Note: the current approach consumes large amount of disk memory. To overcome this problem, computing skipthought embedding during training is a possible solution. However, this will slow down the training process. Some cache mechanism might help to solve this problem.

### Step 2 Train decoder
Set storyteller_training_config = number of training data in stv_text_code.tfrecord Then run:
```
python train.py --input_file_name "TRAIN_DATA_PATH/stv_text_code.tfrecord"
                --input_vocab "TRAIN_DATA_PATH/vocab_shared.txt"
                --pretrained_embedding "TRAIN_DATA_PATH/embeddings_r.npy"
                --train_dir "MODEL_CHECKPOINT_PATH"
