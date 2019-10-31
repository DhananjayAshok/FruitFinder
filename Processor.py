"""
Will have
1. Normalization and other basic data processing
2. Data Augmentation Function

"""
# Processor
#region
def normalize(data):
    return data/255

def flatten(data, height=300, width=200):
    data = data.reshape(-1, height * width * 3)
    return data

def encode(labels):
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import os
    root = "fruits"
    le = LabelEncoder().fit(labels)
    np.save(os.path.join(root, 'classes.npy'), le.classes_)
    return le.transform(labels)

def labelled_process(data, labels):
    data = normalize(data)
    return data, encode(labels)

def unlabelled_process(data):
    data = normalize(data)
    return data
#endregion

# Splitter
#region
def labelled_split(data, labels):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2) # Took out stratification
    print("Split First")
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    print("Split Second")
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def unlabelled_split(data):
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(data, test_size=0.2)
    X_train, X_valid = train_test_split(X_train, test_size=0.2)
    return X_train, X_valid, X_test

#endregion

# Augmentation
#region
def augment(X_train):
    """
    """
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    aug.fit(X_train)
    return aug
#endregion

# Final Assembly
#region
def labelled_final():
    import numpy as np
    import os
    root = "fruits"
    flag = "X_train.npy" in os.listdir(root)
    if flag:
        X_train = np.load(os.path.join(root, "X_train.npy"))
        X_test = np.load(os.path.join(root, "X_test.npy"))
        X_valid = np.load(os.path.join(root, "X_valid.npy"))
        y_train = np.load(os.path.join(root, "y_train.npy"))
        y_test = np.load(os.path.join(root, "y_test.npy"))
        y_valid = np.load(os.path.join(root, "y_valid.npy"))
        return X_train, X_valid, X_test, y_train, y_valid, y_test, augment(X_train)
    else:
        from Loader import load_labelled_data
        data, labels = load_labelled_data()
        print("Loaded")
        data, labels = labelled_process(data, labels)
        print("Preprocessing")
        X_train, X_valid, X_test, y_train, y_valid, y_test = labelled_split(data, labels)
        np.save(os.path.join(root, "X_train"), X_train)
        np.save(os.path.join(root, "X_test"), X_test)
        np.save(os.path.join(root, "X_valid"), X_valid)
        np.save(os.path.join(root, "y_train"), y_train)
        np.save(os.path.join(root, "y_test"), y_test)
        np.save(os.path.join(root, "y_valid"), y_valid)
        print("Split and Saved")
        return X_train, X_valid, X_test, y_train, y_valid, y_test, augment(X_train)
    return

def unlabelled_final():
    from Loader import load_unlabelled_data
    data = load_unlabelled_data()
    data = unlabelled_process(data)
    X_train, X_test = unlabelled_split(data)
    return X_train, X_valid, X_test, augment(X_train)

#endregion

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test, aug = labelled_final()
    print(X_train.shape)