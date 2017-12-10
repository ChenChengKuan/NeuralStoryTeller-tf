# NeuralStoryTeller-tf

This project aims to reproduce the awesome work [Neural Story Teller](https://github.com/ryankiros/neural-storyteller). Since the original work by Kiros was written in Theano which won't be mainted in the future. This project attempts to borrow the ideas from the original work and replace them with tensorflow. This repo is still an ongoing work but can be used to generate simple result.

# Background

Kiros and Samin had provided great explanations of how Neural Story Teller work. Please refer to original [repo](https://github.com/ryankiros/neural-storyteller) and Samin's [blog post](https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed) so I just give a wrap up here. Basically, it contains for modules as shown in the figure below:
![model overview](https://github.com/ChenChengKuan/NeuralStoryTeller-tf/blob/master/imgs/neuralstoryteller.png)
* 1.Image to captions

    This is used to extract textual content of an image. The original implementation used visual semantic embedding [1] to extract 100 (default) captions of the images.
    
* 2.Skipt-thought vectors[2]:

    After the captions are extracted from the images, the final textual representation is computed by taking avearge of the skip-thought vector of captions.
    
* 3.story decoder

   To generate stories, we need to a decoder to help us decode the sentence from the skip-thought embedding space to text. The input and output of decoder are **tokenized text passage** of desired style (Ex. fairy tale, romantic novel) and the correspinding indexlied token (Ex: Input: I have a lovely cat, Output: [0,2,3,10,11] ). The text  passage first encoded by skip-thougut model which serve as the initial state of decoder then the decoder try to learn to decode the passage correctly.

* 4.Style-shfit

   Style shift is a bridge between our source target skipt-thought embedding and the target one. As reported by the original work, text encoded by skipthought vector has the analogy relationship like word2vec (i.e king - man + women = queen). Style shift applies similar idea by minus the original style c (red) and add target style b (purple) to convert the original skipthought vector to the target domain.
   
Finally, the final story is generated by applying beam search to the decoder and choose the one with th less probability.
   
# Implementation schedule and plan

* 1. Image to caption: I plan to use show and tell [3] but the overhead of reproduce this model is large (hardware issue). Therefore, I use the original one and refactor it for easier usage.
* Skip-thought vector: I used the tensorflow [official implementation](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)
* story decoder: I provided the training code with arbitrary corpus. Please refer to decoder for more details


# Result

    

    



