# project-3-dunna21_parsonr6
Ryan Parsons
Andrew Dunn

11-17-2020
CSCI 497/597

Enhancement selection: Stereo evaluation
Our implementation of stereo evaluation takes place by running the stereo_eval.py file
with an argument passed as one of the Middlebury datasets (in the same manner as
plane_sweep_stereo.py). The output of the given file is the Root Mean Square error and the 
bad matching pixels percentage, as provided by Scharstein and Szeliski. This was accomplished by first 
converting our calculated depth map to disparity, then using the metrics outlined in the paper.

We did find it necessary to translate 'inf' values in the
ground truth to the next highest disparity value, but we believe that this has not diminished
the quality of our results as they remain terrible in spite of this. Other implementation details
concern the use of util.py's pyrdown method to handle resizing the images to corresponding dimensions as
well as dataset.py's load_dataset method for importing the data.

Results:
 ________________________________________________________
| Dataset:       | RMSE:         | Bad match %:          |
|________________|_______________|_______________________|
| Flowers        | 330.26908     | 0.99910               |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
