#ifndef HT_STFT_FRAME_WINDOW_TILING_H
#define HT_STFT_FRAME_WINDOW_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HtStftFrameWindowTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inputLength);
    TILING_DATA_FIELD_DEF(uint32_t, frameLength);
    TILING_DATA_FIELD_DEF(uint32_t, hopLength);
    TILING_DATA_FIELD_DEF(uint32_t, frameCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HtStftFrameWindow, HtStftFrameWindowTilingData)
}
#endif
