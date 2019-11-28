#ML-HTML CODE GENERATOR
from os import listdir
from numpy import array
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tkinter as tk
from tkinter import filedialog
from tkinter import font


images = []
all_filenames = listdir('/home/tarun/Documents/PYTHON PROJECT/PYTHON-ML-20191123T101120Z-001/PYTHON-ML/Python_Web/HTML/images/')
all_filenames.sort()
for filename in all_filenames:
    images.append(img_to_array(load_img('/home/tarun/Documents/PYTHON PROJECT/PYTHON-ML-20191123T101120Z-001/PYTHON-ML/Python_Web/HTML/images/'+filename, target_size=(299, 299))))
images = np.array(images, dtype=float)
images = preprocess_input(images)

IR2 = InceptionResNetV2(weights='imagenet', include_top=False)
features = IR2.predict(images)

max_caption_len = 100
tokenizer = Tokenizer(filters='', split=" ", lower=False)

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

X = []
all_files = listdir('/home/tarun/Documents/PYTHON PROJECT/PYTHON-ML-20191123T101120Z-001/PYTHON-ML/Python_Web/HTML/html/')
all_files.sort()
for filename in all_files:
    X.append(load_doc('/home/tarun/Documents/PYTHON PROJECT/PYTHON-ML-20191123T101120Z-001/PYTHON-ML/Python_Web/HTML/html/'+filename))

tokenizer.fit_on_texts(X)

vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(X)
max_length = max(len(s) for s in sequences)

X, y, image_data = list(), list(), list()
for img_no, seq in enumerate(sequences):
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        image_data.append(features[img_no])
        X.append(in_seq[-100:])
        y.append(out_seq)

X, y, image_data = np.array(X), np.array(y), np.array(image_data)

image_features = Input(shape=(8, 8, 1536,))
image_flat = Flatten()(image_features)
image_flat = Dense(128, activation='relu')(image_flat)
ir2_out = RepeatVector(max_caption_len)(image_flat)

language_input = Input(shape=(max_caption_len,))
language_model = Embedding(vocab_size, 200, input_length=max_caption_len)(language_input)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = TimeDistributed(Dense(128, activation='relu'))(language_model)

decoder = concatenate([ir2_out, language_model])
decoder = LSTM(512, return_sequences=False)(decoder)
decoder_output = Dense(vocab_size, activation='softmax')(decoder)

from keras.models import model_from_json
json_file = open('/home/tarun/Documents/PYTHON PROJECT/PYTHON-ML-20191123T101120Z-001/PYTHON-ML/Python_Web/HTML/model2_600epochs.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/home/tarun/Documents/PYTHON PROJECT/PYTHON-ML-20191123T101120Z-001/PYTHON-ML/Python_Web/HTML/model2_600epochs.h5")
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

in_text=''

def generate_desc(model, tokenizer, photo, max_length):
    global in_text
    in_text = 'START'
    for i in range(900):
        sequence = tokenizer.texts_to_sequences([in_text])[0][-100:]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        print(' ' + word, end='')
        if word == 'END':
            break
    return


# Tkinter Front End 
filename=''

def exit(event):
    root.destroy()

def UploadAction(event=None):
    global filename
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    x='Selected : '+filename
    txt.delete(0.0,tk.END)
    txt.insert(0.0,x)
    Label_func("Waiting")   

def RunAction(event=None):
    #print('Running:', filename)
    x="Running: "+filename
    txt.delete(0.0,tk.END)
    txt.insert(0.0,x)
    Label_func("Finished Running")   
    
    test_image = img_to_array(load_img(filename, target_size=(299, 299)))
    test_image = np.array(test_image, dtype=float)
    test_image = preprocess_input(test_image)
    test_features = IR2.predict(np.array([test_image]))
    generate_desc(model, tokenizer, np.array(test_features), 100)
    x=in_text.replace("START","",1)
    x=x[::-1]
    x=x.replace("DNE","",1)
    x=x[::-1]
    txt.delete(0.0,tk.END)
    txt.insert(0.0,x)
    
def BackAction(event=None):
    txt.delete(0.0,tk.END)
    Label_func("Upload Image")
    
def DownloadAction(event=None):
    filename='rand'
    x=txt.get(1.0,tk.END)
    filename=filedialog.asksaveasfilename(defaultextension=".html", filetypes=(("html file", "*.html"),))
    with open(filename,'w') as f:
        f.write(x)

def Label_func(s):
    lbl1.config(text=s)
    
root = tk.Tk()
root.geometry("1600x1200")
root.config(background="black")
root.bind("<Escape>", exit)

Hel=font.Font(family='song ti', size=16, weight='bold')
HelTitle=font.Font(family='song ti', size=38, weight='bold')

btn = tk.Button(root, text='Upload', command=UploadAction,relief=tk.SUNKEN)
btn.config(height=2,width=8,font=Hel,fg="black",bg="#4dbd6b",padx=12,pady=12)
btn.place(relx=0.25,rely=0.2)   

run = tk.Button(root, text='Run', command=RunAction,relief=tk.SUNKEN)
run.config(height=2,width=8,font=Hel,fg="black",bg="#4dbd6b",padx=12,pady=12)
run.place(relx=0.40,rely=0.20)

dl = tk.Button(root, text='Download', command=DownloadAction,relief=tk.SUNKEN)
dl.config(height=2,width=8,font=Hel,fg="black",bg="#4dbd6b",padx=12,pady=12)
dl.place(relx=0.55,rely=0.2)

back = tk.Button(root, text='Back',command=BackAction,relief=tk.SUNKEN)
back.config(height=2,width=8,font=Hel,fg="black",bg="#4dbd6b",padx=12,pady=12)
back.place(relx=0.70,rely=0.2)


lbl=tk.Label(root,text='Web Design Generator')
lbl.config(height=2,width=20,font=HelTitle,fg="#4dbd6b",bg="black")
lbl.place(relx=0.39,rely=0.05)

txt=tk.Text(root)
txt.config(font=Hel,bg="black",fg="#4dbd6b",padx=12,pady=12)
txt.config(height=30,width=98)  
txt.place(relx=0.24,rely=0.35)

lbl2=tk.Label(root,text='CODE STATUS:')
lbl2.config(height=2,width=15,font=Hel,fg="#4dbd6b",bg="black")
lbl2.place(relx=0.04,rely=0.3)

lbl1=tk.Label(root,text='') 
lbl1.config(height=2,width=17,font=Hel,fg="#4dbd6b",bg="black")
lbl1.place(relx=0.12,rely=0.3)


root.mainloop()
