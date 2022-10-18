<h2>EfficientNet-Prostate-Cancer (Updated: 2022/10/19)</h2>
<a href="#1">1 EfficientNetV2 Prostate Cancer Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Prostate-Cancer dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Prostate-Cancer Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Prostate Cancer Classification</a>
</h2>

 This is an experimental EfficientNetV2 Prostate Cancer Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>

This image dataset used here has been taken from the following website:<br>
https://github.com/MicheleDamian/prostate-gleason-dataset<br><br>

<a href="https://github.com/MicheleDamian/prostate-gleason-dataset/blob/master/LICENSE.md">DATASET LICENCE</a><br>
About Prostate Gleason Dataset: Excerpt from the website above:<br>
<div>

Prostate Gleason Dataset is an image database of prostate cancer biopsies derived from the PANDA dataset. The database consists of about 70K patches of size 256x256 pixels extracted from the PANDA's biopsies with overlap. The patches are grouped in 4 classes depending on their Gleason grade:
</div>
Class 0: Stroma and Gleason 0<br>
Class 1: Gleason 3<br>
Class 2: Gleason 4<br>
Class 3: Gleason 5<br>
<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<br>

<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Prostate-Cancer.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Prostate-Cancer
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        ├─Prostate_Images
        └─test
</pre>
<h3>
<a id="1.2">1.2 Prepare Prostate_Cancer dataset</a>
</h3>

The image dataset used here has been taken from the following website:<br>
<a href="https://github.com/MicheleDamian/prostate-gleason-dataset">MicheleDamian/prostate-gleason-dataset</a>
<br>
<pre>
Prostate_Images
├─test
│  ├─0
│  ├─1
│  ├─2
│  └─3
└─train
    ├─0
    ├─1
    ├─2
    └─3    
</pre>

</pre>
The number of images in the dataset is the following:<br>
<img src="./projects/Prostate-Cancer/_Prostate_Images_.png" width="740" height="auto"><br>
<br>

Prostate_Images/train/Cyst:<br>
<img src="./asset/Prostate_Images_train_0.png" width="840" height="auto">
<br>
<br>
Prostate_Images/train/Normal:<br>
<img src="./asset/Prostate_Images_train_1.png" width="840" height="auto">
<br>
<br>
Prostate_Images/train/Stone:<br>
<img src="./asset/Prostate_Images_train_2.png" width="840" height="auto">
<br>
<br>
Prostate_Images/train/Tumor:<br>
<img src="./asset/Prostate_Images_train_3.png" width="840" height="auto">
<br>


<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Prostate-Cancer Classification</a>
</h2>
We have defined the following python classes to implement our Prostate-Cancer Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-b0</b> model to train Prostate-Cancer Classification FineTuning Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b0.tgz">efficientnetv2-b0.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-b0
└─projects
    └─Prostate-Cancer
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Prostate-Cancer Classification efficientnetv2 model by using
<a href="./projects/Prostate-Cancer/Prostate_Cancer_Images/train">Resampled_Kidney_Simpler_Disease_Images/train</a>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-b0  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-b0/model ^
  --optimizer=rmsprop ^
  --image_size=224 ^
  --eval_image_size=224 ^
  --data_dir=./Prostate_Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --num_epochs=50 ^
  --batch_size=8 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = Fale
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 20
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.2, 2.0]
data_format        = "channels_last"

[validation]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 20
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.2, 2.0]
data_format        = "channels_last"

</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Prostate-Cancer/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Prostate-Cancer/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Prostate_Cancer_train_at_epoch_25_1019.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Prostate-Cancer/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Prostate-Cancer/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the breast cancer in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-b0  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --image_path=./test/*.jpeg ^
  --eval_image_size=480 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
0
1
2
3
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Prostate-Cancer/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Prostate-Cancer/Prostate_Images/test">Prostate_Imagess/test</a>.
<br>
<img src="./asset/Prostate_Images_test.png" width="840" height="auto"><br>

<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Prostate-Cancer/inference/inference.csv">inference result file</a>.
<br>
<br>
Inference console output:<br>
<img src="./asset/Prostate_Cancer_infer_at_epoch_25_1019.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Prostate_Cancer_inference_at_epoch_25_1019.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Prostate-Cancer/Prostate_Images/test">
Prostate_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-b0  ^
  --model_dir=./models ^
  --data_dir=./Prostate_Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --eval_image_size=224 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Prostate-Cancer/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Prostate-Cancer/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Prostate_Cancer_evaluate_at_epoch_25_1019.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Prostate_Cancer_classificaiton_report_at_epoch_25_1019.png" width="740" height="auto"><br>

<br>
Confusion matrix:<br>
<img src="./projects/Prostate-Cancer/evaluation/confusion_matrix.png" width="740" height="auto"><br>


<br>
<h3>
References
</h3>
<b>1. Prostate Gleason Dataset</b><br>
<pre>
https://github.com/MicheleDamian/prostate-gleason-dataset<br><br>
</pre>

