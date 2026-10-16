#include "ht_stft_frame_window_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t CHANNELS = 2;
constexpr uint32_t INPUT_LENGTH = 351232;
constexpr uint32_t FRAME_LENGTH = 4096;
constexpr uint32_t HOP_LENGTH = 1024;
constexpr uint32_t FRAME_COUNT = (INPUT_LENGTH - FRAME_LENGTH) / HOP_LENGTH + 1;
static_assert(FRAME_COUNT == 340, "Unexpected number of frames");
static_assert((FRAME_COUNT - 1) * HOP_LENGTH + FRAME_LENGTH == INPUT_LENGTH,
              "The last frame must end exactly at the input boundary");

// Fixed-shape teaching version: launch one core and process all 680 frames.
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* x = context->GetInputShape(0);
    const auto* w = context->GetInputShape(1);
    const auto* y = context->GetOutputShape(0);
    if (x == nullptr || w == nullptr || y == nullptr) return ge::GRAPH_FAILED;
    const auto& xs = x->GetOriginShape();
    const auto& ws = w->GetOriginShape();
    const auto& ys = y->GetOriginShape();
    if (xs.GetDimNum() != 2 || xs.GetDim(0) != CHANNELS || xs.GetDim(1) != INPUT_LENGTH)
        return ge::GRAPH_FAILED;
    if (ws.GetDimNum() != 1 || ws.GetDim(0) != FRAME_LENGTH)
        return ge::GRAPH_FAILED;
    // ACL callers must allocate the full output, not the stage-1 input size.
    if (ys.GetDimNum() != 3 || ys.GetDim(0) != CHANNELS ||
        ys.GetDim(1) != FRAME_COUNT || ys.GetDim(2) != FRAME_LENGTH)
        return ge::GRAPH_FAILED;

    HtStftFrameWindowTilingData tiling;
    tiling.set_channels(CHANNELS);
    tiling.set_inputLength(INPUT_LENGTH);
    tiling.set_frameLength(FRAME_LENGTH);
    tiling.set_hopLength(HOP_LENGTH);
    tiling.set_frameCount(FRAME_COUNT);
    context->SetBlockDim(1);
    context->SetTilingKey(1);  // Must match TILING_KEY_IS(1) in the kernel.
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;
    y->SetDimNum(3);
    y->SetDim(0, optiling::CHANNELS);
    y->SetDim(1, optiling::FRAME_COUNT);
    y->SetDim(2, optiling::FRAME_LENGTH);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class HtStftFrameWindow : public OpDef {
public:
    explicit HtStftFrameWindow(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        // Caller supplies a periodic Hann window already multiplied by 1/64.
        this->Input("window").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtStftFrameWindow);
}
