#include "kernel_operator.h"

class KernelHtStftFrameWindow {
public:
    __aicore__ inline KernelHtStftFrameWindow() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR window, GM_ADDR y, uint32_t channels,
        uint32_t inputLength, uint32_t frameLength, uint32_t hopLength, uint32_t frameCount) {
        channels_ = channels;
        inputLength_ = inputLength;
        frameLength_ = frameLength;
        hopLength_ = hopLength;
        frameCount_ = frameCount;
        xGm_.SetGlobalBuffer((__gm__ float*)x, channels_ * inputLength_);
        windowGm_.SetGlobalBuffer((__gm__ float*)window, frameLength_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, channels_ * frameCount_ * frameLength_);
        // Three 16 KiB buffers: frame input, fixed window, and frame output.
        pipe_.InitBuffer(inQ_, 1, frameLength_ * sizeof(float));
        pipe_.InitBuffer(windowQ_, 1, frameLength_ * sizeof(float));
        pipe_.InitBuffer(outQ_, 1, frameLength_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        // Load the 4096 coefficients once. Keep them in UB for all 680 frames.
        auto windowLocal = windowQ_.AllocTensor<float>();
        AscendC::DataCopy(windowLocal, windowGm_, frameLength_);
        windowQ_.EnQue(windowLocal);
        windowLocal = windowQ_.DeQue<float>();
        for (uint32_t channel = 0; channel < channels_; ++channel) {
            for (uint32_t frame = 0; frame < frameCount_; ++frame) {
                // Input layout: [channel, time].
                const uint32_t inputOffset = channel * inputLength_ + frame * hopLength_;
                // Output layout: [channel, frame, sample_inside_frame].
                const uint32_t outputOffset = (channel * frameCount_ + frame) * frameLength_;
                ProcessFrame(inputOffset, outputOffset, windowLocal);
            }
        }
        windowQ_.FreeTensor(windowLocal);
    }

private:
    __aicore__ inline void ProcessFrame(uint32_t inputOffset, uint32_t outputOffset,
        const AscendC::LocalTensor<float>& windowLocal) {
        // GM -> UB input queue.
        auto in = inQ_.AllocTensor<float>();
        AscendC::DataCopy(in, xGm_[inputOffset], frameLength_);
        inQ_.EnQue(in);

        // Vector computation: windowLocal already contains Hann[n] / 64.
        in = inQ_.DeQue<float>();
        auto out = outQ_.AllocTensor<float>();
        AscendC::Mul(out, in, windowLocal, frameLength_);
        outQ_.EnQue(out);
        inQ_.FreeTensor(in);

        // UB -> GM; neighboring frames have separate output storage.
        out = outQ_.DeQue<float>();
        AscendC::DataCopy(yGm_[outputOffset], out, frameLength_);
        outQ_.FreeTensor(out);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQ_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> windowQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQ_;
    AscendC::GlobalTensor<float> xGm_, windowGm_, yGm_;
    uint32_t channels_, inputLength_, frameLength_, hopLength_, frameCount_;
};

extern "C" __global__ __aicore__ void ht_stft_frame_window(
    GM_ADDR x, GM_ADDR window, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelHtStftFrameWindow op;
        op.Init(x, window, y, tilingData.channels, tilingData.inputLength,
                tilingData.frameLength, tilingData.hopLength, tilingData.frameCount);
        op.Process();
    }
}
