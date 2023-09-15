Embedding Fingerprints into Deep Neural Networks
====
Implementing concepts from two research papers, specifically "Embedding Watermarks into Deep Neural Networks" [1] 
and "DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models" [2], 
the code embeds a digital fingerprint into a wide residual network  throughout the training process. 
The embedding is achieved by applying a parameter regularizer to the target layer.

The code has been revised to accommodate TensorFlow version 2.12.0. 
For further reference, you can access the original project through the following link: https://github.com/yu4u/dnn-watermark

## Requirements
```sh
pip install tensorflow==2.12.0 
pip install scikit-learn==1.2.2 
pip install tensorflow-model-optimization==0.7.5
```

## Embedding Process
Embed the fingerprint during the training phase of the host network:

```sh
python train_model.py config/train_random.json
```

Train the host network *without* embedding:

```sh
python train_model.py config/train_non.json 
```

Compare the fingerprints from the embedded and the non-embedded model:

```sh
python utility/check_embedded_fingerprint.py 
```

Visualising the distribution of the fingerprint extraction:

![](images/fingerprint.png)

## Attacks:

Prune the fingerprinted layer while fine-tuning the model:

```sh
python attacks/pruning_attack.py config/finetune.json
```

Compare the fingerprints from the pruned and the original model:

```sh
python utility/extract_pruned_fingerprint.py 
```

Below are the results of the model accuracy for various sparsity levels within the target layer:

![](images/model_accuracy_sparsity_levels.png)

<!--
<div style="text-align: left;">
    <img src="images_new/fingerprint_sparsity0.1.png" alt="Image 1" style="display: inline-block; margin: 0 auto; width: 400px; height: 300px;">
    <img src="images_new/fingerprint_sparsity0.5.png" alt="Image 2" style="display: inline-block; margin: 0 auto; width: 400px; height: 300px;">
    <img src="images_new/fingerprint_sparsity0.9.png" alt="Image 2" style="display: inline-block; margin: 0 auto; width: 400px; height: 300px;">
</div>
-->
## License
All code in this repository is protected by copyright law and is provided for specific usage outlined below. 
By using this code, you agree to adhere to the following guidelines:

1. **Mandatory Referencing**: You are required to acknowledge and reference this project in your work whenever you use 
or adapt any code from this repository.

2. **Permitted Usage**: This code is intended for research and educational purposes only. 
Any other use requires explicit written permission from the copyright holder.

3. **No Warranty**: This code is provided without any warranties or guarantees. 
The copyright holder is not responsible for any damages or issues resulting from its use.

For questions, permissions, or inquiries related to this project, please open an issue in the 
[GitHub Issue Tracker](https://github.com/AlexandraPosa/fingerprint-embedding-wrn/issues). 
Thank you for your cooperation and adherence to these guidelines.

## References
[1] Y. Uchida, Y. Nagai, S. Sakazawa, and S. Satoh, "Embedding Watermarks into Deep Neural Networks", ICMR, 2017. \
[2] Huili Chen, Bita Darvish Rouhani, Cheng Fu, Jishen Zhao, and Farinaz Koushanfar, "DeepMarks: A Secure Fingerprinting 
    Framework for Digital Rights Management of Deep Learning Models", ICMR, 2019
