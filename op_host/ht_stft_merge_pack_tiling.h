#ifndef HT_STFT_MERGE_PACK_TILING_H
#define HT_STFT_MERGE_PACK_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(HtStftMergePackTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, frequencyBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, batches);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtStftMergePack,HtStftMergePackTilingData)
}
#endif
