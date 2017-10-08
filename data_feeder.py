import os
import pandas as pd
import numpy as np
import cv2
import sklearn

class DataFeeder():
    """
    Class used to preprocess the images and 
    feed them to the network model
    """
    
    def __init__(self, data_dir = "data", batch_size = 128):
        
        self.logfile = os.path.join(data_dir, "driving_log.csv")
        self.batch_size = batch_size
        self.db_index = self._read_index()
        self._list_campaigns()
        self.nsamples = len(self.db_index)
        self.steps_per_epoch = self.nsamples // self.batch_size
        
    def _read_index(self):
        """
        Returns the driving log in pandas dataframe format
        Image paths are converted to relative paths
        """
        filename = self.logfile
        newpath = "IMG"
        
        df = pd.read_csv(filename)
        df.to_csv(filename.replace(".csv", ".bkp"), index = False)
        
        for col in ["right", "center", "left"]:
            for token in ["\\", "/"]:
                df[col] = df[col].str.split(token).str[-1]
                
            df[col] = df[col].apply(lambda x: os.path.join(newpath, x))

        # Saving the file with the correct data path. Used to 
        # clean up the file structure to run on multiple
        # computers
        df.to_csv(filename, index = False)
        return df
    
    def _list_campaigns(self):
        """
        Tags the datasets according to their collection
        campaigns
        """
        df = self.db_index
        
        # Timestamping the lines
        col = "center"
        token = "_"
        df["timestamp"] = df[col].str.split(token).str[1:-2]
        df["timestamp"] = df["timestamp"].apply(lambda x: "".join(x))
        df["timestamp"] = pd.to_datetime(df["timestamp"], format = "%Y%m%d%H%M")
        
        # Finally tagging the data collection campaigns
        df["campaign"] = df['timestamp'].diff().fillna(10*60*1E9).astype("timedelta64[m]")
        df["campaign"] = df["campaign"].apply(lambda x: 1 if (x > 2) else 0)
        df["campaign"] = df["campaign"].cumsum()
        
        self.db_index = df
        
    def fetch(self):
        """
        Generator function to feed data to the model without
        exhausting the computer memory.
        OBS:
            Just small changes from the generator lesson
        """
    
        while 1: # Loop forever so the generator never terminates

            for offset in range(0, self.nsamples, self.batch_size):
                batch_samples = self.db_index.loc[offset:offset + self.batch_size]

                images = []
                angles = []
                for _, batch_sample in batch_samples.iterrows():
                    center_image = cv2.imread("data/" + batch_sample["center"])
                    center_angle = float(batch_sample["steering"])
                    images.append(center_image)
                    angles.append(center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)    