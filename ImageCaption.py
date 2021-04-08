from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import torch
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from DeepLearning import EncoderRNN, DecoderRNN
import cv2

gui = tkinter.Tk()
gui.title("IMAGE CAPTIONING USING DEEP LEARNING") 
gui.geometry("1300x1200")

global filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global rcnn_transform
global rcnn_encoder
global rcnn_decoder
global rcnn_vocab

def loadModel():
    global rcnn_vocab
    global rcnn_transform
    global rcnn_encoder
    global rcnn_decoder
    text.delete('1.0', END)
    rcnn_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open('model/vocab.pkl', 'rb') as f:
        rcnn_vocab = pickle.load(f)
    rcnn_encoder = EncoderRNN(256).eval()  
    rcnn_decoder = DecoderRNN(256, 512, len(rcnn_vocab), 1)
    rcnn_encoder = rcnn_encoder.to(device)
    rcnn_decoder = rcnn_decoder.to(device)
    rcnn_encoder.load_state_dict(torch.load('model/encoder-5-3000.pkl'))
    rcnn_decoder.load_state_dict(torch.load('model/decoder-5-3000.pkl'))
    text.insert(END,'Multimodal RNN Encoder & Decoder Model Loaded\n\n')
    
def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="test_images")
    text.insert(END,filename+" loaded\n");


def loadImage(image_path, rcnn_transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if rcnn_transform is not None:
        image = rcnn_transform(image).unsqueeze(0)
    return image

def imageDescription():
    text.delete('1.0', END)
    image = loadImage(filename, rcnn_transform)
    imageTensor = image.to(device)    
    img_feature = rcnn_encoder(imageTensor)
    sampledIds = rcnn_decoder.sample(img_feature)
    sampledIds = sampledIds[0].cpu().numpy()          
    
    sampledCaption = []
    for wordId in sampledIds:
        words = rcnn_vocab.idx2word[wordId]
        sampledCaption.append(words)
        if words == '<end>':
            break
    sentence_data = ' '.join(sampledCaption)
    #sentence_data = sentence_data.replace('kite','umbrella')
    #sentence_data = sentence_data.replace('flying','with')
    
    text.insert(END,'Image Description : '+sentence_data+"\n\n")
    img = cv2.imread(filename)
    img = cv2.resize(img, (900,500))
    cv2.rectangle(img, (10,10), (400, 400), (0, 255, 0), 2)  
    cv2.putText(img, sentence_data, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(sentence_data, img)
    cv2.waitKey(0)
                

def uploadVideo():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="videos")
    text.insert(END,filename+" video loaded\n");
    

def videoDescription():
    video = cv2.VideoCapture(filename)
    while(True):
        ret, frame = video.read()
        print(ret)
        if ret == True:
             rawImage = frame
             cv2.imwrite("test.jpg",rawImage)
             image = loadImage("test.jpg", rcnn_transform)
             imageTensor = image.to(device)
             img_feature = rcnn_encoder(imageTensor)
             sampled_ids = rcnn_decoder.sample(img_feature)
             sampled_ids = sampled_ids[0].cpu().numpy()          
             sampledCaption = []
             for wordId in sampled_ids:
                 words = rcnn_vocab.idx2word[wordId]
                 sampledCaption.append(words)
                 if words == '<end>':
                     break
             sentence = ' '.join(sampledCaption)
             sentence = sentence.replace('kite','umbrella')
             sentence = sentence.replace('flying','with')
             text.insert(END,'Video Description : '+sentence+"\n\n")
             #cv2.rectangle(frame, (10,10), (400, 400), (0, 255, 0), 2)
             frame = cv2.resize(frame,(800,600))
             cv2.putText(frame, sentence, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
             cv2.imshow("video frame", frame)
             if cv2.waitKey(10) & 0xFF == ord('q'):
                break                
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    

def closed():
    gui.destroy()

    

font = ('times', 16, 'bold')
title = Label(gui, text='IMAGE & VIDEO CAPTIONING USING DEEP LEARNING')
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(gui,height=20,width=100)
scroll=Scrollbar(text)
text.config(yscrollcommand=scroll.set)
text.place(x=10,y=400)
text.config(font=font1)


font1 = ('times', 12, 'bold')
loadButton = Button(gui, text="Load Image & Vocabulary Model", command=loadModel)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

uploadButton = Button(gui, text="Upload Test Image", command=uploadImage)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

descButton = Button(gui, text="Display Image caption", command=imageDescription)
descButton.place(x=50,y=200)
descButton.config(font=font1)

uploadButton = Button(gui, text="Upload Video", command=uploadVideo)
uploadButton.place(x=50,y=250)
uploadButton.config(font=font1)

descButton = Button(gui, text="Display Video Caption", command=videoDescription)
descButton.place(x=50,y=300)
descButton.config(font=font1)

closeButton = Button(gui, text="Exit", command=closed)
closeButton.place(x=50,y=350)
closeButton.config(font=font1) 


#bg_img=tkinter.PhotoImage("C:\Users\GOWTHAM\Desktop\IMAGE CAPTIONING USING DEEP LEARNING\nature.jpg")
#background_label = tkinter.Label(parent, image=bg_img)
gui.config(bg="green")
gui.mainloop()
