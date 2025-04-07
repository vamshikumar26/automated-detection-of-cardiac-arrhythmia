from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import webbrowser

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import pickle

from keras.layers import  MaxPooling2D
from keras.layers import  Activation, Flatten
from keras.layers import Convolution2D

main = tkinter.Tk()
main.title("Automated Detection of Cardiac Arrhythmia using Recurrent Neural Network")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test, pca
global model, dataset
global filename
global X, Y
accuracy = []
precision = []
recall = []
fscore = []
sensitivity = []
specificity = []

labels = ['Normal heart', 'Ischemic changes (Coronary Artery Disease)', 'Old Anterior Myocardial Infarction',
          'Old Inferior Myocardial Infarction', 'Sinus tachycardy', 'Sinus bradycardy', 'Right bundle branch block']
    
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('279').size()
    label.plot(kind="bar")
    plt.show()

def preprocessDataset():
    global X, Y, dataset, pca
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset.fillna(0, inplace = True)
    dataset["279"] = pd.Series(le.fit_transform(dataset["279"].astype(str)))
    temp = dataset.values
    X = temp[:,0:temp.shape[1]-1]
    Y = temp[:,temp.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(Y)
    X = normalize(X)

    pca = PCA(n_components = 40)
    X = pca.fit_transform(X)
    Y = to_categorical(Y)
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size = 0.2)
    
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Different diseases found in dataset\n\n")
    text.insert(END,str(labels)+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"Total records used to train LSTM & CNN : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test LSTM & CNN  : "+str(X_test.shape[0])+"\n")   


def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    sensitivity.append(se)
    specificity.append(sp)
    text.update_idletasks()    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()     

def runLSTM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, sensitivity, specificity
    text.delete('1.0', END)
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    sensitivity.clear()
    specificity.clear()
    if os.path.exists('model/lstm_model.json'):
        with open('model/lstm_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm = model_from_json(loaded_model_json)
        json_file.close()
        lstm.load_weights("model/lstm_model_weights.h5")
        lstm._make_predict_function()
    else:
        lstm_model = Sequential()#defining deep learning sequential object
        #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
        lstm_model.add(LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
        #adding dropout layer to remove irrelevant features
        lstm_model.add(Dropout(0.2))
        #adding another layer
        lstm_model.add(Dense(100, activation='relu'))
        #defining output layer for prediction
        lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
        #compile LSTM model
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #start training model on train data and perform validation on test data
        hist = lstm_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
        #save model weight for future used
        lstm_model.save_weights('model/lstm_model_weights.h5')
        model_json = lstm_model.to_json()
        with open("model/lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(lstm.summary())
    predict = lstm.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("LSTM", predict, testY)


def runCNN():
    global X, Y
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size = 0.2)
    if os.path.exists('model/cnn_model.json'):
        with open('model/cnn_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()
        cnn.load_weights("model/cnn_model_weights.h5")
        cnn._make_predict_function()
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn.fit(X_train, y_train, batch_size=16, epochs=100, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        cnn.save_weights('model/cnn_model_weights.h5')            
        model_json = cnn.to_json()
        with open("model/cnn_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn.summary())
    predict = cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("CNN", predict, testY)        

def graph():
    f = open('model/lstm_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    lstm_accuracy = data['accuracy']
    lstm_loss = data['loss']

    f = open('model/cnn_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    cnn_accuracy = data['accuracy']
    cnn_loss = data['loss']
    
    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Error Rate')
    plt.plot(lstm_accuracy, 'ro-', color = 'green')
    plt.plot(lstm_loss, 'ro-', color = 'blue')
    plt.plot(cnn_accuracy, 'ro-', color = 'orange')
    plt.plot(cnn_loss, 'ro-', color = 'red')
    plt.legend(['LSTM Accuracy', 'LSTM Loss','CNN Accuracy','CNN Loss'], loc='upper left')
    plt.title('LSTM Vs CNN Training Accuracy & Loss Graph')
    plt.show()

def performanceTable():
    output = '<table border=1 align=center>'
    output+= '<tr><th>Dataset Name</th><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th><th>Sensitivity</th><th>Specificity</th></tr>'
    output+='<tr><td>MIT-BH Dataset</td><td>LSTM</td><td>'+str(accuracy[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td><td>'+str(sensitivity[0])+'</td><td>'+str(specificity[0])+'</td></tr>'
    output+='<tr><td>MIT-BH Dataset</td><td>CNN</td><td>'+str(accuracy[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td><td>'+str(sensitivity[1])+'</td><td>'+str(specificity[1])+'</td></tr>'
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Automated Detection of Cardiac Arrhythmia using Recurrent Neural Network')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Arrhythmia Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=50,y=200)
lstmButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=50,y=250)
cnnButton.config(font=font1)

graphButton = Button(main, text="LSTM & CNN Training Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

ptButton = Button(main, text="Performance Table", command=performanceTable)
ptButton.place(x=50,y=350)
ptButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
