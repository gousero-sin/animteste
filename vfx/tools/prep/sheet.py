import sys, cv2, numpy as np
# usage: sheet.py dir t0 t1 step out [w]
d=sys.argv[1]; t0,t1,st=map(float,sys.argv[2:5]); out=sys.argv[5]; W=int(sys.argv[6]) if len(sys.argv)>6 else 180
FPS=1359000/45431
ts=np.arange(t0,t1+1e-6,st); ims=[]
for t in ts:
    i=min(2264,int(round(t*FPS)))
    import glob
    f=sorted(glob.glob(f'{d}/{i:04d}.*'))[0]
    im=cv2.imread(f); h=int(im.shape[0]*W/im.shape[1]); im=cv2.resize(im,(W,h),interpolation=cv2.INTER_AREA)
    cv2.putText(im,f'{t:.2f}',(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1,cv2.LINE_AA)
    ims.append(im)
cols=min(10,len(ims)); rows=(len(ims)+cols-1)//cols
h=ims[0].shape[0]; sheet=np.zeros((rows*h,cols*W,3),np.uint8)
for k,im in enumerate(ims): sheet[(k//cols)*h:(k//cols)*h+h,(k%cols)*W:(k%cols)*W+W]=im
cv2.imwrite(out,sheet)
