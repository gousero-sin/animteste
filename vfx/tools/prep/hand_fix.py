"""Restore the frame-right hand resting on the sink (5.3-7.5 s) that RVM dropped from the matte."""
import sys, os, numpy as np, cv2
sys.path.insert(0,'/home/user/animteste')
from vfx import assets as A
S=sys.argv[1]; dry=len(sys.argv)>2
plate=cv2.imread(f'{S}/plates/plate0.png').astype(np.float32)
for i in range(160,226):
    h=A.hand(i,'L'); p=A.pose(i)
    if h is not None:
        pts=h[:,:2]
    else:
        pts=p[[15,17,19,21],:2]
        c=pts.mean(0); pts=np.concatenate([pts, c+(pts-c)*1.6])
    hull=cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    m=np.zeros((1440,1080),np.uint8); cv2.fillConvexPoly(m,hull,255)
    # include the wrist-forearm junction toward the elbow
    el=p[13,:2]; wr=p[15,:2]; v=wr-el; 
    cv2.line(m,tuple(np.int32(wr-v*0.25)),tuple(np.int32(wr)),255,28)
    m=cv2.dilate(m,np.ones((15,15),np.uint8))
    fr=cv2.imread(f'{S}/hyb/{i:04d}.jpg').astype(np.float32)
    a_old=cv2.imread(f'{S}/alpha/{i:04d}.png',0).astype(np.float32)/255
    lab=cv2.cvtColor((fr/255).astype(np.float32),cv2.COLOR_BGR2Lab)
    # skin model from the opaque forearm near the wrist
    yy,xx=np.mgrid[0:1440,0:1080]
    near=((xx-wr[0])**2+(yy-wr[1])**2<160**2)&(a_old>0.99)
    smp=lab[near]
    if len(smp)>200:
        mu=smp.mean(0); cov=np.cov(smp.T)+np.eye(3)*4.0; ic=np.linalg.inv(cov)
        dd=lab-mu; md=np.einsum('...i,ij,...j->...',dd,ic,dd)
        pskin=np.exp(-0.5*md/2.5)
    else:
        pskin=np.zeros((1440,1080),np.float32)
    pskin=cv2.morphologyEx(pskin.astype(np.float32),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    pskin=cv2.morphologyEx(pskin,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    reg=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),3)*np.clip(pskin*1.6,0,1)
    reg=cv2.GaussianBlur(reg,(0,0),1.2)
    # ramp in/out over the window to avoid pops
    w=min(1.0,(i-160)/6,(225-i)/6)
    src=f'{S}/alpha/{i:04d}.png'
    a=cv2.imread(src,0).astype(np.float32)/255
    an=np.maximum(a,np.clip(reg*1.15,0,1)*w)
    if dry:
        y0,y1,x0,x1=1050,1350,600,900
        o=np.concatenate([fr[y0:y1,x0:x1].astype(np.uint8),cv2.cvtColor((a[y0:y1,x0:x1]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR),cv2.cvtColor((an[y0:y1,x0:x1]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)],1)
        if i in (180,195,206,213): cv2.imwrite(f'{S}/dev/hand_{i}.jpg',o)
    else:
        cv2.imwrite(src+'.tmp.png',(an*255+0.5).astype(np.uint8)); os.replace(src+'.tmp.png',src)
print('ok')
