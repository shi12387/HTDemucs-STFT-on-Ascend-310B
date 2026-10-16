#include "kernel_operator.h"

class KernelHtStftMergePack {
public:
    __aicore__ inline KernelHtStftMergePack() {}
    __aicore__ inline void Init(GM_ADDR hh,GM_ADDR hl,GM_ADDR lh,GM_ADDR ll,GM_ADDR y,
                               uint32_t blocks,uint32_t batches) {
        blocks_=blocks;batches_=batches;
        const uint32_t count=blocks_*2*batches_*32*48;
        hhGm_.SetGlobalBuffer((__gm__ float*)hh,count);hlGm_.SetGlobalBuffer((__gm__ float*)hl,count);
        lhGm_.SetGlobalBuffer((__gm__ float*)lh,count);llGm_.SetGlobalBuffer((__gm__ float*)ll,count);
        yGm_.SetGlobalBuffer((__gm__ float*)y,4*2048*336);
        pipe_.InitBuffer(hhQ_,1,1536*sizeof(float));pipe_.InitBuffer(hlQ_,1,1536*sizeof(float));
        pipe_.InitBuffer(lhQ_,1,1536*sizeof(float));pipe_.InitBuffer(llQ_,1,1536*sizeof(float));
        pipe_.InitBuffer(outQ_,1,336*sizeof(float));
        pipe_.InitBuffer(bandBuf_,704*8*sizeof(float));
        pipe_.InitBuffer(extractIdxBuf_,256*sizeof(uint32_t));
        pipe_.InitBuffer(timeIdxBuf_,2*336*sizeof(uint32_t));
        // Total UB allocation: 52,160 bytes. No 704x48 full-band UB allocation.
        auto extract=extractIdxBuf_.Get<uint32_t>();
        auto times=timeIdxBuf_.Get<uint32_t>();
        for (uint32_t i=0;i<256;++i) extract.SetValue(i,((i/8)*48+i%8)*sizeof(float));
        for (uint32_t c=0;c<2;++c) for (uint32_t t=0;t<336;++t)
            times.SetValue(c*336+t,(c*340+t+2)*8*sizeof(float));
        // Index constants were written by the scalar pipe; Gather reads them on V.
        const auto event=pipe_.FetchEventID(AscendC::HardEvent::S_V);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
    }
    __aicore__ inline void Process() {
        auto band=bandBuf_.Get<float>();
        auto extract=extractIdxBuf_.Get<uint32_t>();
        auto times=timeIdxBuf_.Get<uint32_t>();
        for (uint32_t block=0;block<blocks_;++block) for (uint32_t part=0;part<2;++part) {
            for (uint32_t start=0;start<48;start+=8) {
                const uint32_t firstBin=block*48+start;
                if (firstBin>=2048) continue; // Exclude Nyquist 2048 and all padding.
                for (uint32_t batch=0;batch<batches_;++batch) {
                    const uint32_t offset=((block*2+part)*batches_+batch)*1536;
                    auto hh=hhQ_.AllocTensor<float>();auto hl=hlQ_.AllocTensor<float>();
                    auto lh=lhQ_.AllocTensor<float>();auto ll=llQ_.AllocTensor<float>();
                    AscendC::DataCopy(hh,hhGm_[offset],1536);hhQ_.EnQue(hh);
                    AscendC::DataCopy(hl,hlGm_[offset],1536);hlQ_.EnQue(hl);
                    AscendC::DataCopy(lh,lhGm_[offset],1536);lhQ_.EnQue(lh);
                    AscendC::DataCopy(ll,llGm_[offset],1536);llQ_.EnQue(ll);
                    hh=hhQ_.DeQue<float>();hl=hlQ_.DeQue<float>();
                    lh=lhQ_.DeQue<float>();ll=llQ_.DeQue<float>();
                    // Preserve the old FP32 parentheses: ((HH+HL)+(LH+LL))/64.
                    AscendC::Add(hh,hh,hl,1536);AscendC::Add(lh,lh,ll,1536);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(hh,hh,lh,1536);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(hh,hh,1.0f/64.0f,1536);AscendC::PipeBarrier<PIPE_V>();
                    // Extract 32x8 columns into the 704x8 tile, in frame-major order.
                    AscendC::Gather(band[batch*256],hh,extract,start*sizeof(float),256);
                    AscendC::PipeBarrier<PIPE_V>();
                    hhQ_.FreeTensor(hh);hlQ_.FreeTensor(hl);lhQ_.FreeTensor(lh);llQ_.FreeTensor(ll);
                }
                for (uint32_t channel=0;channel<2;++channel) for (uint32_t j=0;j<8;++j) {
                    auto out=outQ_.AllocTensor<float>();
                    // Select frames 2..337 per channel and transpose to frequency-major.
                    AscendC::Gather(out,band,times[channel*336],j*sizeof(float),336);
                    AscendC::PipeBarrier<PIPE_V>();
                    outQ_.EnQue(out);out=outQ_.DeQue<float>();
                    const uint32_t dst=((channel*2+part)*2048+firstBin+j)*336;
                    AscendC::DataCopy(yGm_[dst],out,336);outQ_.FreeTensor(out);
                }
            }
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> hhQ_,hlQ_,lhQ_,llQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> outQ_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bandBuf_,extractIdxBuf_,timeIdxBuf_;
    AscendC::GlobalTensor<float> hhGm_,hlGm_,lhGm_,llGm_,yGm_;
    uint32_t blocks_,batches_;
};
extern "C" __global__ __aicore__ void ht_stft_merge_pack(
    GM_ADDR hh,GM_ADDR hl,GM_ADDR lh,GM_ADDR ll,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(tilingData,tiling);
    if (TILING_KEY_IS(1)) {
        KernelHtStftMergePack op;
        op.Init(hh,hl,lh,ll,y,tilingData.frequencyBlocks,tilingData.batches);
        op.Process();
    }
}
