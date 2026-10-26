import sys, time, torch, numpy as np, cv2
sys.path.insert(0, sys.argv[1])
from srvgg import load
S=sys.argv[2]
for dn in [0.5, 1.0]:
    m = load(dn, S+'/models')
    im = cv2.imread(f'{S}/allframes/0060.png')[:,:,::-1].astype(np.float32)/255
    x = torch.from_numpy(im.copy()).permute(2,0,1)[None]
    t=time.time()
    with torch.no_grad(): y = m(x)
    print(dn, 'time', time.time()-t)
    y = y[0].permute(1,2,0).clamp(0,1).numpy()
    y = cv2.resize(y, (1080,1440), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{S}/sr_{dn}.png', (y[:,:,::-1]*255).round().astype(np.uint8))
lz = cv2.resize(cv2.imread(f'{S}/allframes/0060.png'), (1080,1440), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(f'{S}/sr_lanczos.png', lz)
