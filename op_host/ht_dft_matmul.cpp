#include "ht_dft_matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>

namespace optiling {
constexpr int32_t M = 32;
constexpr int32_t N = 48;
constexpr int32_t K = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* a = context->GetInputShape(0);
    const auto* b = context->GetInputShape(1);
    const auto* c = context->GetOutputShape(0);
    if (a == nullptr || b == nullptr || c == nullptr || context->GetPlatformInfo() == nullptr)
        return ge::GRAPH_FAILED;
    const auto& as = a->GetOriginShape();
    const auto& bs = b->GetOriginShape();
    const auto& cs = c->GetOriginShape();
    if (as.GetDimNum() != 2 || as.GetDim(0) != M || as.GetDim(1) != K ||
        bs.GetDimNum() != 2 || bs.GetDim(0) != K || bs.GetDim(1) != N ||
        cs.GetDimNum() != 2 || cs.GetDim(0) != M || cs.GetDim(1) != N) {
        std::fprintf(stderr, "HtDftMatmul requires A[32,4096], B[4096,48], C[32,48].\n");
        return ge::GRAPH_FAILED;
    }

    // Obtain the actual target's memory/core properties. No 910B assumptions.
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling api(platform);
    api.SetDim(1);
    api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT16, false);
    api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT16, false);
    api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT);
    api.SetOrgShape(M, N, K);
    api.SetShape(M, N, K);
    api.SetSingleShape(M, N, K);
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);

    HtDftMatmulTilingData tiling;
    const int ret = api.GetTiling(tiling.cubeTiling);
    if (ret == -1) {
        std::fprintf(stderr, "HtDftMatmul: Matmul GetTiling failed for this platform.\n");
        return ge::GRAPH_FAILED;
    }
    // All output must belong to the one core launched by this teaching version.
    if (tiling.cubeTiling.get_usedCoreNum() != 1 ||
        tiling.cubeTiling.get_singleCoreM() < M ||
        tiling.cubeTiling.get_singleCoreN() < N ||
        tiling.cubeTiling.get_singleCoreK() < K) {
        std::fprintf(stderr, "HtDftMatmul: unexpected core partition.\n");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(1);
    context->SetTilingKey(1);
    auto* raw = context->GetRawTilingData();
    if (raw->GetCapacity() < tiling.GetDataSize()) return ge::GRAPH_FAILED;
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    const size_t workspaceSize = platform.GetLibApiWorkSpaceSize();
    context->GetWorkspaceSizes(1)[0] = workspaceSize;
    std::fprintf(stderr, "HtDftMatmul tiling: M=%d N=%d K=%d, cores=1, base=(%d,%d,%d), workspace=%zu\n",
                 M, N, K, static_cast<int>(tiling.cubeTiling.get_baseM()),
                 static_cast<int>(tiling.cubeTiling.get_baseN()),
                 static_cast<int>(tiling.cubeTiling.get_baseK()), workspaceSize);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* c = context->GetOutputShape(0);
    if (c == nullptr) return GRAPH_FAILED;
    c->SetDimNum(2);
    c->SetDim(0, optiling::M);
    c->SetDim(1, optiling::N);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class HtDftMatmul : public OpDef {
public:
    explicit HtDftMatmul(const char* name) : OpDef(name) {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(HtDftMatmul);
}
