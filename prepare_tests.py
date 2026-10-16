#!/usr/bin/env python3
"""Generate deterministic inputs and an independent STFT reference; no model assets."""
import argparse,json
import numpy as np
from support import ROOT,CASES,INPUT_SHAPE,OUTPUT_SHAPE,require,info,baseline,sources

def reference(x,window,backend):
    if backend=='torch':
        import torch
        # Input ALREADY contains any desired padding: center must remain False.
        z=torch.stft(torch.from_numpy(x),n_fft=4096,hop_length=1024,win_length=4096,
                     window=torch.from_numpy(window),center=False,normalized=True,onesided=True,return_complex=True)
        z=z[:,:2048,2:338]
        y=torch.view_as_real(z).permute(0,3,1,2).contiguous().reshape(1,4,2048,336).numpy()
    else:
        frames=np.lib.stride_tricks.sliding_window_view(x,4096,axis=-1)[:,::1024,:]
        require(frames.shape==(2,340,4096),'Incorrect framing')
        z=np.fft.rfft((frames*window).astype(np.float64),n=4096,axis=-1,norm='ortho')[:,2:338,:2048]
        y=np.stack((z[0].real.T,z[0].imag.T,z[1].real.T,z[1].imag.T),axis=0)[None]
    return np.ascontiguousarray(y,dtype='<f4')

def make_cases(seed):
    n=np.arange(351232,dtype=np.float64);phase=2*np.pi*n/4096
    tones=np.stack((.3*np.cos(5*phase)+.12*np.sin(57*phase)+.04*np.cos(2031*phase),
                    .17*np.sin(11*phase)-.09*np.cos(81*phase)+.03*np.sin(2029*phase))).astype('<f4')
    impulse=np.zeros(INPUT_SHAPE,dtype='<f4')
    for index,value in [(0,.25),(3073,1),(65539,-.7),(175617,.4),(351231,-.2)]:impulse[0,index]=value
    for index,value in [(2049,-.5),(8203,.8),(231007,-.3),(350003,.6)]:impulse[1,index]=value
    rng=np.random.default_rng(seed);noise=(rng.standard_normal(INPUT_SHAPE)*.1).astype('<f4');noise[1]*=.37
    return {'tones_warmup':tones,'impulse':impulse,'tones':tones,'noise':noise,'zero':np.zeros(INPUT_SHAPE,dtype='<f4'),'tones_restore':tones}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',choices=('torch','numpy'),default='torch');p.add_argument('--seed',type=int,default=2026);a=p.parse_args();baseline()
    d=ROOT/'test_data';d.mkdir(exist_ok=True)
    if a.reference=='torch':
        try:import torch
        except ImportError as e:raise RuntimeError('Use your original CPU Torch environment, or explicitly select --reference numpy') from e
        window=torch.hann_window(4096,periodic=True,dtype=torch.float32).numpy();version=torch.__version__
    else:
        window=(.5-.5*np.cos(2*np.pi*np.arange(4096,dtype=np.float64)/4096)).astype('<f4');version=np.__version__
    files={}
    for name,x in [('window_hann_f32.bin',window),('window_scaled_f32.bin',(window/np.float32(64)).astype('<f4'))]:x.tofile(d/name);files[name]=info(d/name)
    cases=make_cases(a.seed);cached=None
    for name in CASES:
        x=cases[name];require(np.isfinite(x).all() and np.max(np.abs(x))<65504,'Unsupported test magnitude')
        x.tofile(d/(name+'_input_f32.bin'))
        if name.startswith('tones') and cached is not None:y=cached
        else:
            y=reference(x,window,a.reference)
            if name.startswith('tones'):cached=y
        y.tofile(d/(name+'_reference_f32.bin'))
        for suffix in ('_input_f32.bin','_reference_f32.bin'):files[name+suffix]=info(d/(name+suffix))
    m={'cases':list(CASES),'seed':a.seed,'reference_backend':a.reference,'reference_version':str(version),'input_shape':list(INPUT_SHAPE),'output_shape':list(OUTPUT_SHAPE),
       'contract':{'n_fft':4096,'hop_length':1024,'periodic_hann':True,'normalized':True,'reference_center':False,'crop_frames':[2,338],'crop_bins':[0,2048],'planes':['left_real','left_imag','right_real','right_imag'],'kernel_window':'Hann/64'},
       'files':files,'sources':sources()}
    (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('[TEST DATA] six cases, '+a.reference+' reference; no historical stages, OM weights or external audio')
if __name__=='__main__':main()
