This code will use a picture, overlay a node network and based off of distances between certain nodes, determine what pose the human is in.
How to Use:

1. Download the .py and the IdentifierImage.jpg (any of the 4 images will work as long as they are renamed to this)
2. Put them both into the same Folder on your PC
3. Run The .py
4. You should now see the output it in the console as a string (for example "T-Pose", "standing, etc.)

------------------------------------------------------------------------------------------------------------------------------------------
Current WIP (ML Model):

DataTrain.py: Goes through the .dat data and procecsses it into .npy data that can then be used to train the model.

TrainModel.py: Goes through the processed training data to train the model (accuracy currently quite bad due to it not being able to work over continues stretches of time)

How to use this yourself: First, download some CSI Data from the google collab and adjust the folder structure to look like this:
CSI_Data
    user1
        actual .dat files lie in here (should not need to change anything)
    user2
         actual .dat files lie in here (should not need to change anything)
    ...

Then put the DataTrain.py in the same place as the CSI_data folder and run it. This should generate an example folder with training data (will prob also take a bit)
Then put TrainModel.py in the same place as the previously generated training data, and run it. This should then give you an output on how accurate the model is
In my testing, model accuracy was ~45%, as the model is currently still WIP.
