#include "ht_stft_split_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t CHANNELS=2, FRAMES=340, LENGTH=4096, BATCHES=22, ROWS_PER_BATCH=32;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* input=context->GetInputShape(0);
    if (input==nullptr) return ge::GRAPH_FAILED;
    const auto& x=input->GetOriginShape();
    if (x.GetDimNum()!=3 || x.GetDim(0)!=CHANNELS || x.GetDim(1)!=FRAMES || x.GetDim(2)!=LENGTH)
        return ge::GRAPH_FAILED;
    for (uint32_t i=0;i<2;++i) {
        const auto* output=context->GetOutputShape(i);
        if (output==nullptr) return ge::GRAPH_FAILED;
        const auto& y=output->GetOriginShape();
        if (y.GetDimNum()!=3 || y.GetDim(0)!=BATCHES || y.GetDim(1)!=ROWS_PER_BATCH || y.GetDim(2)!=LENGTH)
            return ge::GRAPH_FAILED;
    }
    HtStftSplitTilingData tiling;
    tiling.set_validRows(CHANNELS*FRAMES);
    tiling.set_paddedRows(BATCHES*ROWS_PER_BATCH);
    tiling.set_rowLength(LENGTH);
    context->SetBlockDim(1);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    for (uint32_t i=0;i<2;++i) {
        auto* y=context->GetOutputShape(i);
        if (y==nullptr) return GRAPH_FAILED;
        y->SetDimNum(3); y->SetDim(0,22); y->SetDim(1,32); y->SetDim(2,4096);
    }
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,ge::DT_FLOAT16);
    context->SetOutputDataType(1,ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class HtStftSplit : public OpDef {
public:
    explicit HtStftSplit(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("high").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("low").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtStftSplit);
}
