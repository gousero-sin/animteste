import sys, numpy as np, cv2, onnxruntime as ort
S=sys.argv[1]; variant=sys.argv[2]
lama=ort.InferenceSession(f'{S}/models/lama_fp32.onnx',providers=['CPUExecutionProvider'])
sr=ort.InferenceSession(f'{S}/models/srvgg_dn05.onnx',providers=['CPUExecutionProvider'])
P=np.array
polys=[P([(39,212),(88,212),(89,352),(60,363),(39,366)]),
       P([(40,213),(130,203),(192,216),(192,226),(115,232),(108,242),(70,242)]),
       P([(44,190),(55,183),(72,184),(80,181),(91,187),(91,212),(44,213)]),
       P([(28,254),(44,254),(44,283),(28,283)]),P([(34,326),(44,326),(44,347),(34,347)]),
       P([(35,355),(57,355),(55,480),(35,480)]),
       P([(10,423),(42,423),(42,453),(10,453)])]
m=np.zeros((480,360),np.uint8)
for p in polys: cv2.fillPoly(m,[p.astype(np.int32)],255)
obj=cv2.dilate(m,np.ones((5,5),np.uint8))
big=obj.copy()
if variant=='strip':
    cv2.fillPoly(big,[P([(0,178),(96,178),(96,370),(62,372),(62,480),(0,480)]).astype(np.int32)],255)
    big=cv2.dilate(big,np.ones((3,3),np.uint8))
p2=cv2.imread(f'{S}/plates/lo_2_flush.png')
x0,y0,x1,y1=0,140,256,480   # crop 256x340 -> scale to 512 tall
crop=p2[y0:y1,x0:x1]; mc=big[y0:y1,x0:x1]
sc=512/(y1-y0); w=int(round((x1-x0)*sc)); w-=w%8
im=cv2.resize(crop,(w,512),interpolation=cv2.INTER_CUBIC)[:,:,::-1].astype(np.float32)/255
mk=(cv2.resize(mc,(w,512),interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)
pad=512-w
im=np.pad(im,((0,0),(0,pad),(0,0)),mode='reflect'); mk=np.pad(mk,((0,0),(0,pad)))
out=lama.run(None,{'image':im.transpose(2,0,1)[None],'mask':mk[None,None]})[0][0].transpose(1,2,0)
if out.max()>2: out=out/255.
out=out[:,:w]
# keep hi-res inpaint: produce plate3 at output res directly from lama result upscaled + SR of low res for the rest
lo=cv2.resize(out,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)[:,:,::-1]
new=p2.copy(); 
mm=cv2.GaussianBlur(big.astype(np.float32)/255,(0,0),2); mm=np.maximum(mm,(big>127).astype(np.float32))
reg=new[y0:y1,x0:x1].astype(np.float32)
new[y0:y1,x0:x1]=(np.clip(lo*255,0,255)*mm[y0:y1,x0:x1,None]+reg*(1-mm[y0:y1,x0:x1,None])).round().astype(np.uint8)
cv2.imwrite(f'{S}/plates/lo_3_shower_{variant}.png',new)
x=new[:,:,::-1].astype(np.float32)/255
y=sr.run(None,{'x':x.transpose(2,0,1)[None].copy()})[0][0].transpose(1,2,0)[:,:,::-1]
hi=cv2.resize(np.clip(y,0,1),(1080,1440),interpolation=cv2.INTER_AREA)
prev=cv2.imread(f'{S}/plates/plate2.png').astype(np.float32)/255
mh=cv2.resize(big,(1080,1440)).astype(np.float32)/255
mh=cv2.GaussianBlur(cv2.dilate(mh,np.ones((25,25),np.uint8)),(0,0),8)[...,None]
hi=hi*mh+prev*(1-mh)
cv2.imwrite(f'{S}/plates/plate3_{variant}.png',(hi*255).round().astype(np.uint8))
cv2.imwrite(f'{S}/plates/mask_shower_{variant}.png',cv2.resize(big,(1080,1440)))
cv2.imwrite(f'{S}/plates/mask_shower_obj.png',cv2.resize(obj,(1080,1440)))
