# id2223_lab2

Group 23: Ernan Wang, Shuyi Chen.

In this lab, we fine-tuned a pre-trained transformer model, Whisper, for the Swedish language. The program is refactored into a feature engineering pipeline and a training pipeline as well as a serverless UI to realize the interaction. All of the codes are implemented on Colab. The model accepts two kinds of input: the user’s speech record through the microphone and the URL of Youtube’s video. The link to the Application webpage on Hugging Face is: https://huggingface.co/spaces/irena/ASR_ID2223 

## The Pre-trained Modal - Whisper
[Whisper](https://huggingface.co/blog/fine-tune-whisper) is a pre-trained model for automatic speech recognition (ASR). It is based on the encoder-decoder model that maps audio spectrogram features to text tokens. Through the pre-training on a large number of labelled audio-transcription data, Whisper can realize better performance with fine-tuning. In this lab, we choose the model of Whisper_small to implement.

## The Feature Pipeline
The [feature_pipeline.ipynb](https://github.com/irena123333/id2223_lab2/blob/main/feature_pipeline.ipynb) is used to extract the features and the Swedish data we used is from the subgroup “sv-SE” of [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), a series of crowd-sourced datasets. The preprocessing of the feature pipeline includes the removal of irrelevant data and extracting the spectrogram features of the audio. As the features we obtained are huge and to fulfil the requirement of the training pipeline, we stored the folder for testing in Hopsworks and the folder for training in Google Drive respectively.

## The Training Pipeline
In [training_pipeline.ipynb](https://github.com/irena123333/id2223_lab2/blob/main/training_pipeline.ipynb) we used GPU on Colab. The pre-trained model Whisper is loaded and the word error rate (WER) is set to evaluate the performance. We configurated max steps as 4000 and steps for checkpoints as 1000. After the 8-hour training, we realized a WER of 19.69%.

## The User Interface
The interactive UI is created in Hugging Face as shown in [app.py](https://github.com/irena123333/id2223_lab2/blob/main/huggingface_space/app.py). We designed two pages supporting different kinds of inputs. The one is transcribing from the recording, through this the user just needs to click the bottom to record and speak in Swedish directly to the microphone. After submitting it will output the text. The other method is to transcribe from a YouTube URL via pasting and submitting the link of a Swedish video. 

### Question: how to improve model performance?
#### Model-centric approach
Utilize larger models, for example, Whisper_large.  
Optimize hyperparameters for training including learning rate and dropout.  
Use a larger pre-trained checkpoint.  
#### Datal-centric approach
Choose advanced feature extraction methods such as MFCC features.  
Use other Swedish data sources, including official or paid databases and updating the model via the data provided by users’ recording.  

