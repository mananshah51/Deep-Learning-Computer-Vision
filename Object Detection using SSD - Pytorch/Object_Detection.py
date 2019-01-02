import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0] #This is the First Transformation where in we convert
    #the frame into right dimensions and color value
    
    #Converting the NumpyArray into Tensor
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    #We use the Permute Function so as to get green, red and blue as it required to NN
    
    #We need Fake Dimensions because NN will not accept a single image or a single vector.
    #It will accept it in Batch Sizes. 
    
    x = Variable(x.unsqueeze(0)) #Therefore we now convert this into Torch Variable. 
    
    y = net(x) #We input the tranformed x variable in the Neural Network 
    
    detections = y.data #Therfore we create a New Tensor because we do not get the output
    #directly from y, therefore we use its data attribute. 
    
    scale = torch.Tensor([width, height, width, height])
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j ,0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,255,0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2 ,cv2.LINE_AA)
            j += 1
    return frame

net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#Object Detection on a Video
reader = imageio.get_reader('4.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('4_Output.mp4', fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print (i)
writer.close()