# Conceptor
Audio Analysis by Conceptor 

# Usage instruction for pattern recognition

```
import conceptor.util as util
import conceptor.reservoir as reservoir
import conceptor.logic as logic

# Create a reservoir network, parameters depending on the feature dimension and task complexity
RNN = reservoir.Reservoir(39, 10, sr = 1.2, in_scale = 0.2, bias_scale = 1)

# Feed a list of data matrix to the reservoir for training, each column of a data matrix is a feature vector
C_list = RNN.recognition_train([femalefeatures, malefeatures])

# testfeatures is the data matrix to be classified, results are the classification results
results, evidence = RNN.recognition_predict(testfeatures, C_list)

```
