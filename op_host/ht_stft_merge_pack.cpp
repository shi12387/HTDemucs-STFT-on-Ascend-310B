#include "ht_stft_merge_pack_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const int64_t dims[]={43,2,22,32,48};
    for (uint32_t i=0;i<4;++i) {
        const auto* input=context->GetInputShape(i);
        if (input==nullptr) return ge::GRAPH_FAILED;
        const auto& x=input->GetOriginShape();
        if (x.GetDimNum()!=5) return ge::GRAPH_FAILED;
        for (uint32_t d=0;d<5;++d) if (x.GetDim(d)!=dims[d]) return ge::GRAPH_FAILED;
    }
    const auto* output=context->GetOutputShape(0);
    if (output==nullptr) return ge::GRAPH_FAILED;
    const auto& y=output->GetOriginShape();
    if (y.GetDimNum()!=4 || y.GetDim(0)!=1 || y.GetDim(1)!=4 || y.GetDim(2)!=2048 || y.GetDim(3)!=336)
        return ge::GRAPH_FAILED;
    HtStftMergePackTilingData tiling;
    tiling.set_frequencyBlocks(43);tiling.set_batches(22);
    context->SetBlockDim(1);context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* y=context->GetOutputShape(0);if (y==nullptr) return GRAPH_FAILED;
    y->SetDimNum(4);y->SetDim(0,1);y->SetDim(1,4);y->SetDim(2,2048);y->SetDim(3,336);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,ge::DT_FLOAT);return GRAPH_SUCCESS;
}
}
namespace ops {
class HtStftMergePack : public OpDef {
public:
    explicit HtStftMergePack(const char* name):OpDef(name) {
        this->Input("hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("hl").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("lh").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("ll").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtStftMergePack);
}
