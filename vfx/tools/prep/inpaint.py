import sys, numpy as np, cv2, onnxruntime as ort
S=sys.argv[1]
lama=ort.InferenceSession(f'{S}/models/lama_fp32.onnx',providers=['CPUExecutionProvider'])
sr=ort.InferenceSession(f'{S}/models/srvgg_dn05.onnx',providers=['CPUExecutionProvider'])
P=np.array
polys={
 'clutter':[P([(240,276),(262,272),(267,276),(267,337),(246,343),(240,340)]),
            P([(222,287),(238,283),(242,290),(240,303),(224,305)]),
            P([(294,309),(308,309),(308,324),(294,324)]),
            P([(243,404),(250,392),(258,383),(262,370),(270,365),(280,368),(285,382),(300,390),(318,392),(326,398),(332,415),(334,470),(330,480),(252,480),(248,430)]),
            P([(198,438),(242,438),(244,480),(196,480)])],
 'flush':[P([(179,364),(194,360),(199,365),(199,390),(184,394),(179,390)])],
 'shower':[P([(39,212),(88,212),(89,352),(60,363),(39,366)]),
           P([(40,213),(130,203),(192,216),(192,226),(115,232),(108,242),(70,242)]),
           P([(44,190),(55,183),(72,184),(80,181),(91,187),(91,212),(44,213)]),
           P([(28,254),(44,254),(44,283),(28,283)]),P([(34,326),(44,326),(44,347),(34,347)]),
           P([(35,355),(57,355),(55,480),(35,480)]),
           P([(10,423),(42,423),(42,453),(10,453)])],
}
def mask_of(name):
    m=np.zeros((480,360),np.uint8)
    for p in polys[name]: cv2.fillPoly(m,[p.astype(np.int32)],255)
    return cv2.dilate(m,np.ones((5,5),np.uint8))
def run_lama(img,mask):
    # img uint8 BGR 480x360, mask uint8. resize to 512 height, pad width to 512
    im=cv2.resize(img,(384,512),interpolation=cv2.INTER_CUBIC)[:,:,::-1].astype(np.float32)/255
    mk=(cv2.resize(mask,(384,512),interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)
    im=np.pad(im,((0,0),(64,64),(0,0)),mode='reflect'); mk=np.pad(mk,((0,0),(64,64)))
    out=lama.run(None,{'image':im.transpose(2,0,1)[None],'mask':mk[None,None]})[0][0].transpose(1,2,0)
    if out.max()>2: out=out/255.
    out=out[:,64:448]
    out=cv2.resize(out,(360,480),interpolation=cv2.INTER_AREA)[:,:,::-1]
    return np.clip(out*255,0,255).astype(np.uint8)
def comp(base,new,mask,feather=3):
    m=cv2.GaussianBlur(mask.astype(np.float32)/255,(0,0),feather)
    m=np.maximum(m,(mask>127).astype(np.float32))[...,None]
    return (new*m+base*(1-m)).round().astype(np.uint8)
def run_sr(img):
    x=img[:,:,::-1].astype(np.float32)/255
    y=sr.run(None,{'x':x.transpose(2,0,1)[None].copy()})[0][0].transpose(1,2,0)[:,:,::-1]
    return cv2.resize(np.clip(y,0,1),(1080,1440),interpolation=cv2.INTER_AREA)
raw=cv2.imread(f'{S}/plates/plate_lo_raw.png'); hole=cv2.imread(f'{S}/plates/plate_hole.png',0)
hole=cv2.dilate(hole,np.ones((5,5),np.uint8))
p0=comp(raw,run_lama(raw,hole),hole,1)
masks={'hole':hole}
seq=[('p0',p0,None)]
cur=p0
for name in ['clutter','flush','shower']:
    m=mask_of(name); masks[name]=m
    new=comp(cur,run_lama(cur,m),m)
    seq.append((name,new,m)); cur=new
his=[]
prev_hi=None
for k,(name,img,m) in enumerate(seq):
    cv2.imwrite(f'{S}/plates/lo_{k}_{name}.png',img)
    hi=run_sr(img)
    if prev_hi is not None:
        mh=cv2.resize(m,(1080,1440),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
        mh=cv2.GaussianBlur(cv2.dilate(mh,np.ones((25,25),np.uint8)),(0,0),8)[...,None]
        hi=hi*mh+prev_hi*(1-mh)
        cv2.imwrite(f'{S}/plates/mask_{name}.png',(np.clip(cv2.resize(m,(1080,1440),interpolation=cv2.INTER_LINEAR),0,255)).astype(np.uint8))
    cv2.imwrite(f'{S}/plates/plate{k}.png',(hi*255).round().astype(np.uint8))
    prev_hi=hi
print('ok')
