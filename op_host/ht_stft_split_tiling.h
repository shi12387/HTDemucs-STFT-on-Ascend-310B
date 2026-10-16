#ifndef HT_STFT_SPLIT_TILING_H
#define HT_STFT_SPLIT_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HtStftSplitTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, validRows);
    TILING_DATA_FIELD_DEF(uint32_t, paddedRows);
    TILING_DATA_FIELD_DEF(uint32_t, rowLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtStftSplit, HtStftSplitTilingData)
}
#endif
