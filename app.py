from models import resnext50_32x4d
from dataset import CassavaDataset, get_transforms, classes
from inference import load_state, inference
from utils import CFG
from grad_cam import SaveFeatures, getCAM, plotGradCAM
from deploy import deploy_app
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from html_mardown import app_off,app_off2, model_predicting, loading_bar, result_pred, image_uploaded_success, more_options, class0, class1, class2, class3, class4, s_load_bar, class0_side, class1_side, class2_side, class3_side, class4_side

#Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

#Set App title
st.title('Cassava disease prediction Web App')


#Set the directory path
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


#Read and display the banner
banner = Image.open(banner_path)
st.sidebar.image(banner,use_column_width=True)

#App description
st.write("The app predicts diseases in Cassava plants. The model was trained with the [cassava leaf disease dataset on Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification/data). ")
st.write('**For more info:** [Blog Post](https://aminey.medium.com/how-to-train-ml-models-with-mislabeled-data-cf4bb353b3d9?sk=9f4ce905cd5c4f2d86ec3bf7b93d024c) **|** **Code:** [Github repository](https://github.com/Amiiney/cld-app-streamlit) **|** **Contact info:** [LinkedIn](https://www.linkedin.com/in/amineyamlahi/)')
st.markdown('***')


#Load the model and the weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnext50_32x4d(CFG.model_name, pretrained=False)
states = [load_state( my_path + '/weights/resnext50_32x4d_fold0_best.pth')]
#model_path= my_path + '/weights/resnext50_32x4d_fold0_best.pth'
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'], strict=True)
#For Grad-cam features
final_conv = model.model.layer4[2]._modules.get('conv3')
fc_params = list(model.model._modules.get('fc').parameters())


#Set the selectbox for demo images
st.write('**Select an image for a DEMO**')
menu = ['Select an Image','Image 1', 'Image 2', 'Image 3']
choice = st.selectbox('Select an image', menu)


#Set the box for the user to upload an image
st.write("**Upload your Image**")
uploaded_image = st.file_uploader("Upload your image in JPG or PNG format", type=["jpg", "png"])


#DataLoader for pytorch dataset
def Loader(img_path, upload_state=False, demo_state=True):
    test_dataset = CassavaDataset(test,img_path, uploaded_image,transform=get_transforms(data='valid'), uploaded_state=upload_state, demo_state=demo_state)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
    return test_loader

test_loader= Loader(img_1_path, upload_state=True, demo_state=False) #Uploaded image
test_loader1= Loader(img_1_path) #Demo Image 1
test_loader2= Loader(img_2_path) #Demo Image 2
test_loader3= Loader(img_3_path) #Demo Image 3


#Set red flag if no image is selected/uploaded
if uploaded_image is None and choice=='Select an Image':
    st.sidebar.markdown(app_off, unsafe_allow_html=True)
    st.sidebar.markdown(app_off2, unsafe_allow_html=True)


#Deploy the model if the user uploads an image
if uploaded_image is not None:
    deploy_app(test_loader, uploaded_image, uploaded=True, demo=False)


#Deploy the model if the user selects Image 1
if choice== 'Image 1':
    deploy_app(test_loader1, img_1_path)


#Deploy the model if the user selects Image 2
if choice== 'Image 2':
    deploy_app(test_loader2, img_2_path) 


#Deploy the model if the user selects Image 3
if choice== 'Image 3':
    deploy_app(test_loader3, img_3_path)  
    


