#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

extern "C" __global__ __aicore__ void ht_dft_matmul(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        GlobalTensor<half> aGm, bGm;
        GlobalTensor<float> cGm;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a), 32 * 4096);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), 4096 * 48);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), 32 * 48);

        // Use the default Norm template; no transposition, bias or atomic add.
        Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTiling);
        mm.SetTensorA(aGm, false);
        mm.SetTensorB(bGm, false);
        mm.SetTail(32, 48, 4096);
        mm.IterateAll(cGm);
        mm.End();
    }
}
