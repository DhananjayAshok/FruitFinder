"""
Will have
1. Timing Class
2. Function to load the unlabelled flower data as numpy arrays ( or call on memory to fetch the array if this isn't the first time its running)
3. Function to load the labelled flower data as numpy arrays ('''')
"""
class Timer:
    """
    Utility class to time the program while running. 
    """
    def __init__(self, start_time):
        self.start_time = start_time
        self.counter = 0

    def time(self, message=None):
        """
        Timing function that returns the time taken for this step since the starting time. Message is optional otherwise we use a counter. 
        """
        if message:
            print(f"{message} at {time.time()-self.start_time}")
        else:
            print(f"{self.counter} at {time.time()-self.start_time}")
            self.counter += 1
        return


def load_unlabelled_data(height=300, width=200):
    pass



def load_labelled_data(height =100, width = 100): # Flowers was 300, 200
    import os
    import cv2
    import numpy as np
    from PIL import Image
    root = os.path.join('fruits', "Data") # Flowers was 'flowers'
    master_root = os.path.join('fruits')
    if "data.npy" not in os.listdir(master_root) and "labels.npy" not in os.listdir(master_root):
        data = []
        labels = []
        corrupt = {}
        directories = os.listdir(root)
        for category in directories:
            for img_path in os.listdir(os.path.join(root, category)):
                try:
                    final_path = os.path.join(root, category, img_path)
                    image = cv2.imread(final_path)
                    image_from_array = Image.fromarray(image, mode="RGB")
                    resized = image_from_array.resize((width, height))
                    final = np.array(resized)
                    data.append(final)
                    labels.append(category)
                except:
                    corrupt[category] = corrupt.get(category, 0) + 1
        data = np.array(data).astype('int32')
        labels = np.array(labels)
        np.save(os.path.join(master_root, "data"), data)
        np.save(os.path.join(master_root, "labels"), labels)
        print(corrupt)
        return data, labels
    else:
        return np.load(os.path.join(master_root, "data.npy")), np.load(os.path.join(master_root, "labels.npy"))





if __name__ == "__main__":
    # Imports
    #region
    import time
    timer = Timer(time.time())
    timer.time()
    data, labels = load_labelled_data()
    timer.time()

    #endregion
