# project-3-dunna21_parsonr6
Ryan Parsons
Andrew Dunn

11-17-2020
CSCI 497/597

Enhancement selection: Stereo evaluation
Our implementation of stereo evaluation takes place by running the stereo_eval.py file
with an argument passed as one of the Middlebury datasets (in the same manner as
plane_sweep_stereo.py). The output of the given file is the Root Mean Square error and the 
bad matching pixels percentage, as provided by Scharstein and Szeliski.

We did find it necessary to translate 'inf' values in the
ground truth to the next highest disparity value, but we believe that this has not diminished
the quality of our results as they remain terrible in spite of this.

Results:
 ________________________________________________________
|Dataset:        | RMSE:         | Bad match %:          |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
|                |               |                       |
|________________|_______________|_______________________|
