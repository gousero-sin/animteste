"""Remove removed-object leakage from the person matte: inside the removal masks, pixels whose
frame colour matches the original bathroom plate (i.e. the object, not the person) get alpha 0."""
import sys, os, numpy as np, cv2, shutil
S=sys.argv[1]; a0,a1=int(sys.argv[2]),int(sys.argv[3]); dry=len(sys.argv)>4
P=f'{S}/plates'
m=np.zeros((1440,1080),np.float32)
for n in ['clutter','flush','shower_region']:
    m=np.maximum(m,cv2.imread(f'{P}/mask_{n}.png',0).astype(np.float32)/255)
m=cv2.dilate((m>0.3).astype(np.uint8),np.ones((17,17),np.uint8)).astype(np.float32)
m=cv2.GaussianBlur(m,(0,0),4)
ys,xs=np.where(m>0.01); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
plate=cv2.imread(f'{P}/plate0.png').astype(np.float32)[y0:y1,x0:x1]
plate_b=cv2.GaussianBlur(plate,(0,0),1.5)
mc=m[y0:y1,x0:x1]
os.makedirs(f'{S}/alpha_rvm',exist_ok=True)
for i in range(a0,a1):
    src=f'{S}/alpha/{i:04d}.png'; bak=f'{S}/alpha_rvm/{i:04d}.png'
    if not os.path.exists(bak): shutil.copy(src,bak)
    a=cv2.imread(bak,0).astype(np.float32)/255
    fr=cv2.imread(f'{S}/hyb/{i:04d}.jpg').astype(np.float32)[y0:y1,x0:x1]
    d=np.abs(cv2.GaussianBlur(fr,(0,0),1.5)-plate_b).max(-1)
    bgp=np.clip((30-d)/16,0,1)
    # never eat the solid interior of the body: only partially-opaque or object-like pixels
    solid=cv2.erode((a[y0:y1,x0:x1]>0.995).astype(np.uint8),np.ones((9,9),np.uint8)).astype(np.float32)
    bgp=bgp*(1-solid)*np.clip((0.96-a[y0:y1,x0:x1])/0.15,0,1)
    bgp=cv2.GaussianBlur(bgp,(0,0),1.0)
    ac=a[y0:y1,x0:x1]*(1-bgp*mc)
    a[y0:y1,x0:x1]=ac
    out=(np.clip(a,0,1)*255+0.5).astype(np.uint8)
    if dry:
        cv2.imwrite(f'{S}/dev/alpha_gate_{i:04d}.png',np.concatenate([cv2.imread(bak,0)[y0:y1,x0:x1],out[y0:y1,x0:x1]],1))
    else:
        cv2.imwrite(src+'.tmp.png',out); os.replace(src+'.tmp.png',src)
print('done',a0,a1)
