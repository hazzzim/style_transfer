import matplotlib.pyplot as plt
import cv2
import numpy as np
import subprocess
import torch
from stylenet import StyleNetwork
from torchvision import transforms as T
from torchvision.utils import save_image
from transformer_net import TransformerNet
import glob


device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
counter_frame =0
counter_model =0

def resize_crop(image):
    return image

def change_model (path,devise):
    state_dict = torch.load(path)
    for k in list(state_dict.keys()):
        if(k.find('running_mean')>0) or (k.find('running_var')>0):
            del state_dict[k]



    torch.cuda.empty_cache()
    net=TransformerNet()
    net.load_state_dict(state_dict)
    net.cuda()
    for p in net.parameters():
    	  p.requires_grad=False

    net.eval()

    torch.cuda.empty_cache()

    #device=torch.device('cpu')

    net=net.eval().to(device) #use eval just for safety

    return net

net=change_model('./models/space.pth',device)

src=cv2.VideoCapture('/dev/video0') #USB camera ID

model_list =[]

for models in glob.glob('./models/'+"*"):
    model_list.append(models)



torch.cuda.empty_cache()


ffstr='ffmpeg -re -f rawvideo -pix_fmt rgb24 -s 640x480 -i - -f v4l2 -pix_fmt yuv420p /dev/video2'
#ffmpeg pipeline which accepts raw rgb frames from command line and writes to virtul camera in yuv420p format

v2out=subprocess.Popen(ffstr, shell=True, stdin=subprocess.PIPE) #open process with shell so we can write to it

while True:
  
    ret, frame=src.read()
    frame=(frame[:,:,::-1]/255.0).astype(np.float32) #convert BGR to RGB, convert to 0-1 range and cast to float32
    frame_tensor=torch.unsqueeze(torch.from_numpy(frame),0).permute(0,3,1,2)
    # add batch dimension and convert to NCHW format
    preprocess=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    tensor_in = preprocess(frame_tensor) #normalize
    tensor_in=tensor_in.to(device) #send to GPU
    tensor_out = net(tensor_in) #stylized tensor
    image=tensor_out[0]
    
    tensor_out=torch.squeeze(tensor_out).permute(1,2,0) #remove batch dimension and convert to HWC (opencv format)
    stylized_frame=((tensor_out.to('cpu').detach().numpy())).astype(np.uint8) #convert to 0-255 range and cast as uint8
    torch.cuda.empty_cache()
  
    v2out.stdin.write(stylized_frame.tobytes())

    torch.cuda.empty_cache()
    
    counter_frame += 1
 
    if counter_frame%40 ==0 :

        net=change_model(model_list[counter_model],device)
        counter_model +=1

        if counter_model ==10:
            counter_model=0


    

  
# closing all open windows
cv2.destroyAllWindows()

