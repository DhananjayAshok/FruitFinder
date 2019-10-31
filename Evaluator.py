"""
Use classification analysis, plot precision, recall and ROC curve
Use the notebook
    https://www.kaggle.com/anktplwl91/visualizing-what-your-convnet-learns
"""

def get_random_images(n=1, categories=None):
    from PIL import Image
    import cv2
    import os
    import numpy as np
    root = os.path.join("fruits","Data")
    width = 100
    height = 100

    if not categories:
        target_folders = os.listdir(root)
    else:
        target_folders = [set(categories).intersection(set( os.listdir(root) ) ) ]
        if target_folders is None:
            target_folders = os.listdir(root)

    data = []
    labels = []
    start = 0
    i = 0
    while start < n:
        try:
            print(f"Trying so far {start} on {i}")
            folder = os.path.join(root, np.random.choice(target_folders)) 
            img_path = np.random.choice(os.listdir(folder))
            final_path = os.path.join(folder, img_path)
            img = cv2.imread(final_path)
            image_from_array = Image.fromarray(img, mode="RGB")
            resized = image_from_array.resize((width, height))
            data.append(np.array(resized).astype('float32')/255)
            labels.append(os.path.basename(folder))
            start+=1
            
        except:
            pass
        i+=1

    return np.array(data), np.array(labels)

def plot_image(data, index, labels):
    try:
        img = data[index].copy()
    except:
        print(f"Index {index} not in data")
        return 

    img[:,:,0] = data[index][:,:,2]
    img[:,:,2]=data[index][:,:,0]
    plt.imshow(img)

    if str(labels.dtype) == "int32":
        labels = _reverse_label_encoding(labels)

    plt.title(labels[index])
    plt.show()
    return

def visualize_conv_layer_ouptut(model, layer_index, X_test, y_test_flat, test_index=0, convolution_index=0):
    from keras.models import Model
    layer_output = model.model.layers[layer_index].output
    activation_model = Model(inputs=model.model.input, outputs=layer_output)
    activations = activation_model.predict(X_test[test_index:test_index+1])
    plot_image(X_test, test_index, y_test_flat)
    img = activations[0, :, :, convolution_index]
    plt.imshow(img, cmap="plasma")
    plt.title(f"{model} - layer {layer_index+1}- filter {convolution_index+1}")
    plt.show()
    return

def _flatten_labels(y):
    import numpy as np
    return np.argmax(y, axis=1)

def _get_classes():
    import numpy as np
    import os
    root = "fruits"
    classes = np.load(os.path.join(root, "classes.npy"))
    return classes

def _reverse_label_encoding(y):
    import numpy as np
    classes = _get_classes()
    final = []
    for item in y:
        final.append(classes[item])
    return np.array(final)

def confusion(y_test_flat, y_preds, display_error=0.5):
    """
    Displays only the probabity of [display_error, 1]
    Implement the checking errors with analysis section
    However there are too many categories for me to directly display it, so I will crop it
    """
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
    y_truths = _reverse_label_encoding(y_test_flat)
    y_pred = _reverse_label_encoding(y_preds)
    classes = _get_classes()
    m = confusion_matrix(y_truths, y_pred, labels=None).astype('float64')
    for col in range(len(m)):
        s = np.sum(m[col])
        m[col] = m[col]/s
    m = pd.DataFrame(m)
    m.columns = classes
    m.set_index(pd.Series(classes), inplace=True)
    return m

def confusion_analysis(m):
    """
    Show the worst 5 categories
    Show the best 5 categories

    """
    def overall(m, top=5):
        acc_list = [m[col][col] for col in m.columns]
        best5 = []
        worst5 = []
        while len(best5) < top:
           best = 0
           worst = 1
           bpos = 0
           wpos = 0
           for n in range(1, len(acc_list)):
               if acc_list[n] >= best:
                   best = acc_list[n]
                   bpos = n
               if acc_list[n] <= worst:
                   worst = acc_list[n]
                   wpos = n
           best5.append((best, m.columns[bpos]))
           worst5.append((worst, m.columns[wpos]))
           if bpos == wpos:
               acc_list.pop(bpos)
           else:
               acc_list.pop(bpos)
               acc_list.pop(wpos)
        return best5, worst5
    best, worst = overall(m)
    print(best)
    print(worst)
    def subcategory(m, top =5):
        """
        First Layer Greedy Approach
        """
        import numpy as np
        def extract_submistake_list(m):
            final = []
            for col in m:
                look = m[1]
                pass
        best = []
        worst = []
        l = extract_submistake_list(m)
        while (len(best)< top):
            
            for item in l:
                pass
        return         
    return 


def accuracy_breakdown(y_test_flat, y_preds):
    # Plot Disply the precision and recall.
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_test_flat, y_preds, average="micro")
    rec = recall_score(y_test_flat, y_preds, average = "micro")
    f = f1_score(y_test_flat, y_preds, average="micro")
    return prec, rec, f



def final_report(model, X_test, y_test):
    y_preds = _flatten_labels(model.predict(X_test))
    y_test_flat = _flatten_labels(y_test)
    data, labels = get_random_images()
    m = confusion(y_test_flat, y_preds)
    confusion_analysis(m)
    p, r, f = accuracy_breakdown(y_test_flat, y_preds)
    print(f"{model} has precision {p*100}%, recall {r*100}% and f1_score {f*100}%")
    if str(model) == "simple_model":
        layer_index = 0
        test_index = 0
        convolutional_index = 0
        visualize_conv_layer_ouptut(model, layer_index, X_test,y_test_flat, test_index, convolutional_index)
    elif str(model) == "residual_model":
        layer_index = 0
        test_index = 0
        convolutional_index = 0
        visualize_conv_layer_ouptut(model, layer_index, X_test,y_test_flat, test_index, convolutional_index)
    elif str(model) == "inception_model":
        layer_index = 0
        test_index = 0
        convolutional_index = 0
        visualize_conv_layer_ouptut(model, layer_index, X_test,y_test_flat, test_index, convolutional_index)
    else:
        layer_index = 0
        test_index = 0
        convolutional_index = 0
        visualize_conv_layer_ouptut(model, layer_index, X_test,y_test_flat, test_index, convolutional_index)
    return


if __name__ == "__main__":
    from Modeller import final_model_generator, final_data_generator
    import matplotlib.pyplot as plt
    import seaborn
    
    model = final_model_generator("residual")
    X_train, X_val, X_test, y_train, y_val, y_test = final_data_generator()
    final_report(model, X_test, y_test)

    
