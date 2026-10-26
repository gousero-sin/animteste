import sys, numpy as np, cv2, glob
S=sys.argv[1]; n=len(glob.glob(f'{S}/alpha_lo/*.png'))-2
idx=list(range(0,n,4))
stack=np.zeros((len(idx),480,360,3),np.float32); wts=np.zeros((len(idx),480,360),np.float32)
for k,i in enumerate(idx):
    stack[k]=cv2.imread(f'{S}/allframes/{i:04d}.png').astype(np.float32)
    a=cv2.imread(f'{S}/alpha_lo/{i:04d}.png',0).astype(np.float32)/255
    a=cv2.dilate(a,np.ones((15,15),np.uint8))
    wts[k]=(a<0.02)
cnt=wts.sum(0)
# masked median: set excluded to nan
st=stack.copy(); st[wts<0.5]=np.nan
med=np.nanmedian(st,axis=0)
hole=(cnt<3).astype(np.uint8)
med=np.nan_to_num(med)
cv2.imwrite(f'{S}/plates/plate_lo_raw.png',med.round().astype(np.uint8))
cv2.imwrite(f'{S}/plates/plate_hole.png',hole*255)
print('frames',len(idx),'hole px',hole.sum())
