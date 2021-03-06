from models import resnext50_32x4d
from dataset import CassavaDataset, get_transforms, classes
from inference import load_state, inference
from utils import CFG
from grad_cam import SaveFeatures, getCAM, plotGradCAM
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from html_mardown import app_off,app_off2, model_predicting, loading_bar, result_pred, image_uploaded_success, more_options, class0, class1, class2, class3, class4, s_load_bar, class0_side, class1_side, class2_side, class3_side, class4_side
import matplotlib

#Setting directory path
github_mode= True

if github_mode:
    my_path= '.'
else:
    my_path= '/Users/amir/Desktop/Projects/Casssava webapp'

test=pd.read_csv( my_path + '/data/sample.csv')
img_1_path= my_path + '/images/img_1.jpg'
img_2_path= my_path + '/images/img_2.jpg'
img_3_path= my_path + '/images/img_3.jpg'
banner_path= my_path + '/images/banner.png'
output_image= my_path + '/images//gradcam2.png'

#Read and load the model and the weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnext50_32x4d(CFG.model_name, pretrained=False)
states = [load_state( my_path+'/weights/resnext50_32x4d_fold0_best.pth')]

#For Grad-cam features
final_conv = model.model.layer4[2]._modules.get('conv3')
fc_params = list(model.model._modules.get('fc').parameters())


#Pytorch DataLoader
def Loader(img_path, upload_state=False, demo_state=True):
    test_dataset = CassavaDataset(test,img_path,transform=get_transforms(data='valid'), uploaded_state=upload_state, demo_state=demo_state)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
    return test_loader

test_loader= Loader(img_1_path, upload_state=True, demo_state=False) #Uploaded image
test_loader1= Loader(img_1_path) #Demo Image 1
test_loader2= Loader(img_2_path) #Demo Image 2
test_loader3= Loader(img_3_path) #Demo Image 3
                       

#Deploy function to be used in app.py
def deploy_app(test_loader, file_path, uploaded=False, demo=True):
    st.markdown('***')
    st.markdown(model_predicting, unsafe_allow_html=True)
    if demo:
        image_1 = cv2.imread(file_path)
    if uploaded:
        image_1 = file_path
    st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
    st.sidebar.image(image_1, width=301, channels='BGR')
    
    for img in test_loader:
        activated_features = SaveFeatures(final_conv)
        #Save weight from fc
        weight = np.squeeze(fc_params[0].cpu().data.numpy())
        #Inference
        logits, output = inference(model, states, img, device)
        pred_idx = output.to('cpu').numpy().argmax(1)
        #Grad-cam image display
        cur_images = img.cpu().numpy().transpose((0, 2, 3, 1))
        heatmap = getCAM(activated_features.features, weight, pred_idx)
        plt.imshow(cv2.cvtColor((cur_images[0]* 255).astype('uint8'), cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.resize((heatmap* 255).astype('uint8'), (328, 328), interpolation=cv2.INTER_LINEAR), alpha=0.4, cmap='jet')
        plt.savefig(output_image)

        
        #Outputing our model's prediction results
        if pred_idx[0]==0:
            st.markdown(class0, unsafe_allow_html=True)
            st.sidebar.markdown(class0_side, unsafe_allow_html=True)
            st.write(" The predicted class is: **Cassava Bacterial Blight (CBB)**" )
        elif pred_idx[0]==1:
            st.markdown(class1, unsafe_allow_html=True)
            st.sidebar.markdown(class1_side, unsafe_allow_html=True)
            st.write("The predicted class is: **Cassava Brown Streak Disease (CBSD)**" )
        elif pred_idx[0]==2:
            st.markdown(class2, unsafe_allow_html=True)
            st.sidebar.markdown(class2_side, unsafe_allow_html=True)
            st.write("The predicted class is: **Cassava Green Mottle (CGM)**" )
        elif pred_idx[0]==3:
            st.markdown(class3, unsafe_allow_html=True)
            st.sidebar.markdown(class3_side, unsafe_allow_html=True)
            st.write("The predicted class is: **Cassava Mosaic Disease (CMD)**" )
        elif pred_idx[0]==4:
            st.markdown(class4, unsafe_allow_html=True)
            st.sidebar.markdown(class4_side, unsafe_allow_html=True)
            st.write("The predicted class is: **Healthy**" )

        st.sidebar.markdown('**Scroll down to read the full report (Grad-cam and class probabilities)**')

        #Display the Grad-Cam image
        st.title('**Grad-cam visualization**')
        st.write('Grad-cam highlights the important regions in the image for predicting the class concept. It helps to understand if the model based its predictions on the correct regions of the image.')
        st.write('*Grad-Cam is facing some color channels conflict. I am working on fixing the bug!*')
        gram_im= cv2.imread(output_image)
        st.image(gram_im, width=528, channels='RGB')
        
        #Display the class probabilities table
        st.title('**Class predictions:**') 
        classes['class probability %']= logits.reshape(-1).tolist()
        classes['class probability %']= classes['class probability %'] * 100
        classes_proba = classes.style.background_gradient(cmap='Reds')
        st.write(classes_proba)