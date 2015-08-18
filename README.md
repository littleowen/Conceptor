# Conceptor
Audio Analysis by Conceptor 

# Usage instruction for pattern recognition

```
import conceptor.recognition as recog

new_recogniser = recog.Recognizer()

new_recogniser.train(training_data)

results = new_recognizer.predict(test_data)

```


# Usage Example for speaker recognitions

See the IPython Notebook:
http://nbviewer.ipython.org/github/littleowen/Conceptor/blob/master/Speaker.ipynb



# Emotion and Tone Recognition

See my blog post for the documentation and usage instructions:
http://reservoir-conceptors.blogspot.kr/2015/07/emotion-recognition-and-tone.html



# Python Speaker Identificaiton Module written for the RedHen Audio Analysis Pipeline
The following changes are made compared with the previous version:
1. Shift from Python3 to Python2
2. Replaced GMM from Sklearn by GMM from PyCASP
3. Added functions to recognize features directly, so that it is ready for the shared features from the pipeline.
4. Return the log likelihood of each prediction so that one can make rejections on untrained classes

Gender identifier as a usage example:

```
import speaker.recognition as SR
Gender = SR.GMMRec() # Create a new recognizer

```

1. Training:
Note: It is highly recommneded to get the training features from audio signals that do not contain any silence, you can use the remove_silence function provided in the module:

```
import scipy.io.wavfile as wav
from speaker.silence import remove_silence
(samplerate, signal) = wav.read(audio_path)
clean_signal = remove_silence(samplerate, signal)
```


```
import numpy as np

# Here we use mfcc as the audio features, but in theory, other audio features should work as well, e.g. lpc
female_mfcc = np.array(get_female_mfcc()) # female_mfcc.shape = (N1, D); N1 vectors and D dimension
male_mfcc = np.array(get_male_mfcc()) # male_mfcc.shape = (N2, D);
Gender.enroll('Female', female_mfcc) # enroll the female audio features
Gender.enroll('Male', male_mfcc) # enroll the male audio features
Gender.train() # train the GMMs with PyCASP
Gender.dump('gender.model') # save the trained model into a file named "gender.model" for future use

```

2. Testing:

```
Gender = SR.GMMRec.load('gender.model') # this is not necessary if you just trained the model
test_mfcc = np.array(get_test_mfcc()) # test_mfcc.shape = (N3, D)
(result, log_lkld) = Gender.predict(test_mfcc) # predict the speaker, where result is the most porbabel speaker label, and log_lkld is the log likelihood for test_mfcc to be from the recognized speaker. 

```


