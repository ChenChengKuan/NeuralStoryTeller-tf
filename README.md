# NeuralStoryTeller-tf

This project aims to reproduce the awesome work [Neural Story Teller](https://github.com/ryankiros/neural-storyteller). Since the original work by Kiros was written in Theano which won't be mainted in the future. This project attempts to borrow the ideas from the original work and replace them with tensorflow. This repo is still an ongoing work but can be used to generate simple result.

## Background

Kiros and Samin had provided great explanations of how Neural Story Teller work. Please refer to original [repo](https://github.com/ryankiros/neural-storyteller) and Samin's [blog post](https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed) so I just give a wrap up here. Basically, it contains for modules as shown in the figure below:

![model overview](https://github.com/ChenChengKuan/NeuralStoryTeller-tf/blob/master/imgs/neuralstoryteller.png)
                                      Overview of Neural Story Teller

* 1.Image to captions

    This is used to extract textual content of an image. The original implementation used visual semantic embedding [1] to extract 100 (default) captions of the images.
    
* 2.Skipt-thought vectors[2]:

    After the captions are extracted from the images, the final textual representation is computed by taking avearge of the skip-thought vector of captions.
    
* 3.story decoder

   To generate stories, we need to a decoder to help us decode the sentence from the skip-thought embedding space to text. The input and output of decoder are **tokenized text passage** of desired style (Ex. fairy tale, romantic novel) and the correspinding indexlied token (Ex: Input: I have a lovely cat, Output: [0,2,3,10,11] ). The text  passage first encoded by skip-thougut model which serve as the initial state of decoder then the decoder try to learn to decode the passage correctly.

* 4.Style-shfit

   Style shift is a bridge between our source (**c**) and target (**b**) skipt-thought embedding. As reported by the original work, text encoded by skipthought vector has the analogy relationship like word2vec (i.e king - man + women = queen). Style shift applies similar idea by minus the original style **c** (red) and add target style **b** (purple) to convert the original skipthought vector to the target domain.
   
Finally, the final story is generated by applying beam search to the decoder and choose the one with th less probability.

## Result
I used the Adventure book as my target style. The story decoder was trained on only 250000 passage samples of Adventure book (very small number compared with the orignal work used 14 million passages of romance novel). It can generate meaningful paragaph with some flaws as shown below. These flaws might be caused by some issues (See To-do part)

<img src="https://github.com/ChenChengKuan/NeuralStoryTeller-tf/blob/master/imgs/universiades_girl.jpeg" height="220px" align="left">

<br>
On the other side of the street , I saw a man standing in the middle of the street . He was sitting in the middle of the living room , wearing a white shirt and a white shirt . He wore a sleeveless white shirt and a pair of jeans and a pair of jeans . A woman with a black hair and a white shirt and a hat on her head was still holding .

## Getting Start
Following steps will guide you how to run demo.ipynb and show the directions of building everything from scratch

### Step 0: Install Required Package
* Tensorflow (>=1.3) ([instructions](https://www.tensorflow.org/install/))
* Theano (1.0.0)([instructions](http://deeplearning.net/software/theano/))
* Numpy ([instructions](https://www.scipy.org/install.html))
* NLTK 
    * First installation ([instructions](http://www.nltk.org/install.html))
    * Download data ([instructions](http://www.nltk.org/data.html))

### Step 1: Prepare caption generator
Actually, you can use any image to caption model given that it can output top K captoins of an input image. This model use visual-semantic embedding as caption generator.

### Step 2: Download the pretrained skipthought (optional but highly recommend)
Follow the offical release skipthout vector to download the [pretrained skipthouht model](https://github.com/tensorflow/models/tree/master/research/skip_thoughts#download-pretrained-models-optional). This repo use uni-skip model only.

### Step 3: Prepare the corpus with target style
You can use any corpus you like given you have large of text. I recommend to download [BookCorpus](http://yknzhu.wixsite.com/mbweb) Dataset (around 5G) if you do not want to prepare corpus. If you want to use customized corpus, please parse it to the following format:

![text foramt](https://github.com/ChenChengKuan/NeuralStoryTeller-tf/blob/master/imgs/text.png)

### Step 4: Train decoder
Please follow the instructions in the [story-decoder](https://github.com/ChenChengKuan/NeuralStoryTeller-tf/tree/master/storyteller).

### Step 5: Style-extractor
If you just want to play with `demo.ipynb`, you can use pre-extracted text style in `style_bias/`. I aslo provide a script `style_extractor.py` to extarct arbitrary corpus style. Just run:
```
python style_extractor.py --stv_vocab "SKIPTHOUGHT_PRETRAINED_PATH/vocab.txt"
                          --stv_embedding "SKIPTHOUGHT_PRETRAINED_PATH/embeddings.npy"
                          --stv_model "SKIPTHOUGHT_PRETRAINED_PATH/model-xxxxxx.ckpt"
                          --book_data_dir "BOOKCORPUS_PATH"
                          --book_data_cat "book_category"
```
Each coprus will extract short and long style. Therefore, there are another two parameters `style_length_cut_long` and `style_length_cut_short`  can be set to decide the threshold of short and long text.


## Implementation schedule and plan

* Image to caption: I plan to use show and tell [3] but the overhead of reproduce this model is large (hardware issue). Therefore, I use the original one and refactor it for easier usage.
* Skip-thought vector: I used the tensorflow [official implementation](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)
* story decoder: I provided the training code with arbitrary corpus. Please refer to decoder for more details

## Issue
* The skipt-thought vector used here will output nan if the length of passage contains more than 610 characters, still not figure the reason
* This repo used single uni-skip representation, which might not be powertful enought to represent the long passages.

## To-do list
- [ ] Replace the uni-skip skipthought embedding with the combination of uni-skip and bi-skip as used in the original work. 
- [ ] Use larger sample to train the story decoder
- [ ] Replace the theano-dependent image to caption model with show and tell

## Reference
[1] Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel. "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models." arXiv preprint arXiv:1411.2539 (2014).<br>
[2] Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. "Skip-Thought Vectors." arXiv preprint arXiv:1506.06726 (2015).<br>
[3]Vinyal et al. "Show and Tell: A Neural Image Caption Generator" CVPR 2015
