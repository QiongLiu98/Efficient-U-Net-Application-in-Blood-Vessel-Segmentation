from PIL import Image
import numpy as np

SN=0
SP=0
ACC=0
TP=0
TN=0
FP=0
FN=0
for i in range(1,21):
    a = Image.open('E:/ziming/unet/unet-master/DRIVE4/test/target/%d.png'%i)
    #b = Image.open('data/test_outputs/densenet121/%d.png'%i)
    b = Image.open('F:/model/test_outputs/unet_png/%d.png'%i)
    a = np.array(a,dtype='int8')
    b = np.array(b,dtype='int8')
   # c = np.zeros((512,512),dtype='int8')
    
    tp = 0
    fn = 0 
    fp = 0
    tn = 0
    for i in range(512):
        for j in range(512):
            if(b[i][j]!=0):
                b[i][j]=1
            if(a[i][j]==1 and b[i][j]==1):
                tp+=1
            if(a[i][j]==1 and b[i][j]==0):
                fn+=1
            if(a[i][j]==0 and b[i][j]==1):
                fp+=1
            if(a[i][j]==0 and b[i][j]==0):
                tn+=1
    
    TP=TP+tp
    TN=TN+tn
    FP=FP+fp
    FN=FN+fn
    sn=tp/(tp+fn)
    sp=tn/(fp+tn)
    acc=(tp+tn)/(tp+tn+fp+fn)
    SN=SN+sn
    SP=SP+sp
    ACC=ACC+acc
TP=TP/20
TN=TN/20
FP=FP/20
FN=FN/20
print('SN=',SN/20)
print('SP=',SP/20)
print('ACC=',ACC/20)    
