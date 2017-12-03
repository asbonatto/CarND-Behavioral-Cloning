import os
import pandas as pd
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DataFeeder():
    """
    Class used to preprocess the images and 
    feed them to the network model
    """
    
    def __init__(self, data_dir = "data", batch_size = 512, test_size = 0.20):
        
        self.logfile = os.path.join(data_dir, "driving_log.csv")
        self.batch_size = batch_size
        self.db_index = self._read_index()
        self.db_index = self._list_campaigns()
        # self.rebalance()
        self.add_side_cameras()
        self.add_flipped()
        
        self.db_train, self.db_valid = train_test_split(self.db_index, test_size = test_size)
        self.db_train.reset_index(inplace = True)
        self.db_valid.reset_index(inplace = True)
        
        self.nsamples = len(self.db_train)
        
        self.steps_per_epoch = self.nsamples // self.batch_size
        self.validation_steps = len(self.db_valid) // self.batch_size
        
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
        
        return df
        
    def describe_campaigns(self):
        """
        Density plots of the data acquisition campaigns
        """
        df = self.db_index

        for campaign in df["campaign"].unique():
            idx = df["campaign"] == campaign
            plt.hist(df.loc[idx, "steering"])
            plt.show()
            
    def add_side_cameras(self, correction = 0.25):
        """
        Extends the datasets with images from the side cameras
        """
        
        correction_dict = {"right": -correction, "left": + correction}
        
        for side in ["right", "left"]:
            df = self.db_index.copy()
            df["center"] = df[side]
            df["steering"]+= + correction_dict[side]
            self.db_index = pd.concat([self.db_index, df], ignore_index = True)
        
    def add_flipped(self):
        """
        Doubles the dataset by marking a flipping algorithm in
        """
        self.db_index["flip"] = 0
        df1 = self.db_index.copy()
        df1["flip"] = 1
        self.db_index = pd.concat([self.db_index, df1], ignore_index = True)
        
        
    
    def rebalance(self, nbins = 40):
        """
        Create bins of approximately the same size for sampling uniformly rebalancing the 
        dataset
        """
        
        
        df = self.db_index
        rebalanced = []
        for campaign in df["campaign"].unique():
            # First create the bins
            idx = df["campaign"] == campaign
            df.loc[idx, "bins"] = pd.qcut(df.loc[idx, "steering"], nbins, duplicates = "drop")
            
            # OLD : subsampling only the central bins
            #bins = sorted(df.loc[idx].bins.unique())
            #central_bins = [bin for bin in bins if bin.left*bin.right <= 0]
            
            groups = df.loc[idx].groupby(["bins"])
            max_samples = int(np.percentile(groups.size(), 80))
            
            resample = groups.filter(lambda x: len(x) >= max_samples)
            rebalanced.append(resample.apply(lambda x: x.sample(n= max_samples).reset_index(drop = True)))
            
            resample = groups.filter(lambda x: len(x) < max_samples)
            rebalanced.append(resample.apply(lambda x: x.sample(frac= 1).reset_index(drop = True)))
            
            
        self.db_index_raw = df    
        self.db_index = pd.concat(rebalanced, ignore_index = True)
        print("Number of samples discarded {}".format(len(self.db_index_raw) - len(self.db_index)))
   
        
    def fetch_train(self):
        """
        Generator function to feed data to the model without
        exhausting the computer memory.
        OBS:
            Just small changes from the generator lesson
        """
        
        while 1: # Loop forever so the generator never terminates
            for offset in range(0, self.nsamples, self.batch_size):
                batch_samples = self.db_train.loc[offset:offset + (self.batch_size - 1)]

                images = []
                angles = []
                for _, batch_sample in batch_samples.iterrows():
                    
                    center_image, center_angle = get_data(batch_sample)
                    
                    images.append(center_image)
                    angles.append(center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def fetch_valid(self):
        """
        Generator function to feed data to the model without
        exhausting the computer memory.
        OBS:
            Just small changes from the generator lesson
        """
        
        while 1: # Loop forever so the generator never terminates
            for offset in range(0, len(self.db_valid), self.batch_size):
                batch_samples = self.db_valid.loc[offset:offset + (self.batch_size - 1)]

                images = []
                angles = []
                for _, batch_sample in batch_samples.iterrows():
                    
                    center_image, center_angle = get_data(batch_sample)
                    
                    images.append(center_image)
                    angles.append(center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

def get_data(batch_sample):
    """
    Helper function to flip the image
    """
    center_image = mpimg.imread("data/" + batch_sample["center"])
    center_angle = float(batch_sample["steering"])
                    
    if batch_sample["flip"] > 0:
        center_image = cv2.flip(center_image, 1)
        center_angle = center_angle*-1.0
    
    return center_image, center_angle

def adjust_brightness(img):
    """
    Performs histogram equalization on brightness channel (HSV)
    """
    
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2[:,:, 2] = clahe.apply(img2[:,:, 2])
    
    return cv2.cvtColor(img2, cv2.COLOR_HSV2RGB)