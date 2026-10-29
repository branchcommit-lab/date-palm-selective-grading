# Authenticated source extraction.
# For audit only: this is not an independently
# executable replacement inference implementation.


# ORIGINAL SYMBOL: Decoder
class Decoder(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        if pretrained:
            net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:net=resnet18(weights=None)
        self.stem=nn.Sequential(net.conv1,net.bn1,net.relu)
        self.pool=net.maxpool
        self.layer1=net.layer1;self.layer2=net.layer2;self.layer3=net.layer3;self.layer4=net.layer4
        def up(a,b,c):return nn.Sequential(nn.Conv2d(a+b,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.ReLU(inplace=True),
                                           nn.Conv2d(c,c,3,padding=1,bias=False),nn.BatchNorm2d(c),nn.ReLU(inplace=True))
        self.up3=up(512,256,256);self.up2=up(256,128,128);self.up1=up(128,64,64);self.up0=up(64,64,48)
        self.head=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,1))
    def forward(self,x):
        a=self.stem(x);b=self.layer1(self.pool(a));c=self.layer2(b);d=self.layer3(c);e=self.layer4(d)
        z=self.up3(torch.cat([F.interpolate(e,size=d.shape[-2:],mode='bilinear',align_corners=False),d],1))
        z=self.up2(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        z=self.up1(torch.cat([F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        z=self.up0(torch.cat([F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return F.interpolate(self.head(z),size=x.shape[-2:],mode='bilinear',align_corners=False)



# ORIGINAL SYMBOL: load_model
def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Decoder(pretrained=False)
    model.load_state_dict(ck['state_dict'], strict=True)
    return model.eval().to(device)



# ORIGINAL SYMBOL: image_read
def image_read(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail(SHAPE, Image.Resampling.BICUBIC)
        canvas=Image.new('RGB',SHAPE,(128,128,128))
        ox=(SHAPE[0]-im.width)//2;oy=(SHAPE[1]-im.height)//2
        canvas.paste(im,(ox,oy))
    return np.array(canvas,dtype=np.uint8)



# ORIGINAL SYMBOL: BenchmarkImageDataset
class BenchmarkImageDataset(torch.utils.data.Dataset):
    """Preserve benchmark order, but decode PNGs in background CPU workers."""
    def __init__(self,samples):
        self.rows=samples.to_dict('records')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        image=image_read(row['image_path'])
        fg=cv2.imread(row['fg_path'],cv2.IMREAD_GRAYSCALE)
        if fg is None:raise RuntimeError('Missing benchmark foreground '+row['fg_path'])
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),
                torch.from_numpy(np.ascontiguousarray(fg>127)),row['sample_id'])



# ORIGINAL SYMBOL: ThreadPrefetchLoader
class ThreadPrefetchLoader:
    """Prefetch CPU batches concurrently with CUDA compute, with no shm.

    The underlying DataLoader has num_workers=0; its sampler and collation are
    unchanged. Only when batches are *prepared* differs. Queue capacity two
    keeps host-memory use bounded. All synthetic random draws are keyed to
    source, seed and epoch, so the examples are identical to the unprefetched
    pipeline. Exception propagation is explicit instead of silently hanging.
    """
    def __init__(self, loader, buffer_batches=2):
        self.loader=loader
        self.buffer_batches=buffer_batches

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        from queue import Queue, Full
        from threading import Thread, Event
        import traceback as _traceback
        q=Queue(maxsize=self.buffer_batches)
        stop=Event()

        def enqueue(kind,payload):
            while not stop.is_set():
                try:
                    q.put((kind,payload),timeout=.2)
                    return
                except Full:
                    continue

        def producer():
            try:


                cv2.setNumThreads(1)
                for batch in self.loader:
                    if stop.is_set():break
                    enqueue('batch',batch)
            except BaseException as exc:
                enqueue('error',(exc,_traceback.format_exc()))
            finally:
                enqueue('done',None)

        worker=Thread(target=producer,name='datepalm-v4-cpu-prefetch',daemon=True)
        worker.start()
        try:
            while True:
                kind,value=q.get()
                if kind=='done':break
                if kind=='error':
                    exc,tb=value
                    raise RuntimeError('CPU preprocessing/prefetch failed:\n'+tb) from exc
                yield value
        finally:
            stop.set()
            worker.join(timeout=2)



# ORIGINAL SYMBOL: v4_loader
def v4_loader(dataset,batch_size,shuffle=False,seed=None):
    """Deterministic, shm-free DataLoader overlapped with GPU computation."""
    settings=dict(batch_size=batch_size,shuffle=shuffle,
                  num_workers=0,pin_memory=V4_PIN_MEMORY)
    if seed is not None:settings['generator']=torch.Generator().manual_seed(seed)
    return ThreadPrefetchLoader(DataLoader(dataset,**settings),buffer_batches=2)



# ORIGINAL SYMBOL: matched_predict
@torch.no_grad()
def matched_predict(ckpt,samples,device,batch_size=24,keep_maps=True,store_path=None):
    model=load_model(ckpt,device);model.eval();result=[]
    ds=BenchmarkImageDataset(samples)
    loader=v4_loader(ds,batch_size,False)


    stored_maps=None
    if keep_maps and store_path is not None:
        from shutil import disk_usage
        store_path=Path(store_path)
        store_path.parent.mkdir(parents=True,exist_ok=True)
        needed=len(ds)*256*256*4
        if disk_usage(store_path.parent).free < needed+2*1024**3:
            raise RuntimeError('Insufficient disk space for bounded matched-model localization')
        stored_maps=np.lib.format.open_memmap(store_path,mode='w+',
                                  dtype=np.float32,shape=(len(ds),256,256))
    processed=0
    live_update(force=True,subtask='Matched benchmark inference (GPU)',
                scoring_samples_done=0,scoring_samples_total=len(ds))
    for images,fg,sample_ids in tqdm(loader,desc='Neural predictions',unit='batch',leave=False):
        xx=images.to(device,non_blocking=True).float().div_(255.)
        maps=torch.sigmoid(model(xx)).float().cpu().numpy()[:,0]
        foreground=fg.numpy()
        for k,sid in enumerate(sample_ids):
            vals=maps[k][foreground[k]]
            if len(vals)<100:raise RuntimeError('Insufficient fruit pixels')
            if keep_maps and stored_maps is not None:
                if maps[k].shape!=(256,256):
                    raise RuntimeError('Unexpected matched-model anomaly map shape')
                stored_maps[processed+k]=maps[k]
            result.append({'sample_id':sid,'score':float(np.quantile(vals,.99)),
                           'map':(stored_maps[processed+k] if stored_maps is not None
                                  else (maps[k] if keep_maps else None))})
        processed+=len(sample_ids)
        live_update(subtask='Matched benchmark inference (GPU)',
                    scoring_samples_done=processed,scoring_samples_total=len(ds))
    if stored_maps is not None:stored_maps.flush()
    del model
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result



# ORIGINAL SYMBOL: score_batch
def score_batch(prob,fg):
    out=[]
    for z,f in zip(prob,fg):
        scores=z[0][f[0]>.5].flatten()
        out.append(float(torch.quantile(scores,.99)) if scores.numel()>100 else float('nan'))
    return out


