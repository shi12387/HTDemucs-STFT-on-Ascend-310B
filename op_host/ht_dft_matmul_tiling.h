#ifndef HT_DFT_MATMUL_TILING_H
#define HT_DFT_MATMUL_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HtDftMatmulTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtDftMatmul, HtDftMatmulTilingData)
}
#endif
