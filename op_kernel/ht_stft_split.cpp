#include "kernel_operator.h"

class KernelHtStftSplit {
public:
    __aicore__ inline KernelHtStftSplit() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR high,GM_ADDR low,
                               uint32_t validRows,uint32_t paddedRows,uint32_t rowLength) {
        validRows_=validRows; paddedRows_=paddedRows; rowLength_=rowLength;
        xGm_.SetGlobalBuffer((__gm__ float*)x,validRows_*rowLength_);
        highGm_.SetGlobalBuffer((__gm__ half*)high,paddedRows_*rowLength_);
        lowGm_.SetGlobalBuffer((__gm__ half*)low,paddedRows_*rowLength_);
        // 16 KiB input + 16 KiB expanded high + two 8 KiB outputs = 48 KiB UB.
        pipe_.InitBuffer(inQ_,1,rowLength_*sizeof(float));
        pipe_.InitBuffer(highQ_,1,rowLength_*sizeof(half));
        pipe_.InitBuffer(lowQ_,1,rowLength_*sizeof(half));
        pipe_.InitBuffer(expandedBuf_,rowLength_*sizeof(float));
    }
    __aicore__ inline void Process() {
        for (uint32_t row=0;row<paddedRows_;++row) {
            auto high=highQ_.AllocTensor<half>();
            auto low=lowQ_.AllocTensor<half>();
            if (row<validRows_) {
                auto x=inQ_.AllocTensor<float>();
                AscendC::DataCopy(x,xGm_[row*rowLength_],rowLength_);
                inQ_.EnQue(x); x=inQ_.DeQue<float>();
                auto expanded=expandedBuf_.Get<float>();
                // Input is the existing Hann/64 output. Recover the unnormalized
                // windowed values before FP16 conversion, exactly as the CPU path.
                AscendC::Muls(x,x,64.0f,rowLength_);
                AscendC::PipeBarrier<PIPE_V>();
                // CAST_NONE uses nearest-even rounding when precision is lost.
                AscendC::Cast(high,x,AscendC::RoundMode::CAST_NONE,rowLength_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(expanded,high,AscendC::RoundMode::CAST_NONE,rowLength_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(x,x,expanded,rowLength_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(low,x,AscendC::RoundMode::CAST_NONE,rowLength_);
                AscendC::PipeBarrier<PIPE_V>();
                inQ_.FreeTensor(x);
            } else {
                // Do not read beyond 680 valid rows; all 24 tail rows are +0.
                AscendC::Duplicate(high,static_cast<half>(0.0f),rowLength_);
                AscendC::Duplicate(low,static_cast<half>(0.0f),rowLength_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            highQ_.EnQue(high); lowQ_.EnQue(low);
            high=highQ_.DeQue<half>(); low=lowQ_.DeQue<half>();
            AscendC::DataCopy(highGm_[row*rowLength_],high,rowLength_);
            AscendC::DataCopy(lowGm_[row*rowLength_],low,rowLength_);
            highQ_.FreeTensor(high); lowQ_.FreeTensor(low);
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> inQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> highQ_,lowQ_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> expandedBuf_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<half> highGm_,lowGm_;
    uint32_t validRows_,paddedRows_,rowLength_;
};
extern "C" __global__ __aicore__ void ht_stft_split(
    GM_ADDR x,GM_ADDR high,GM_ADDR low,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(tilingData,tiling);
    if (TILING_KEY_IS(1)) {
        KernelHtStftSplit op;
        op.Init(x,high,low,tilingData.validRows,tilingData.paddedRows,tilingData.rowLength);
        op.Process();
    }
}
