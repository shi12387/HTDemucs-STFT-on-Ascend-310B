#pragma once
#include "acl/acl.h"
#include "aclnn_ht_stft_frame_window.h"
#include "aclnn_ht_stft_split.h"
#include "aclnn_ht_dft_matmul.h"
#include "aclnn_ht_stft_merge_pack.h"
#include "dft_plan.h"
extern "C" uint32_t HtStftStandalonePreload_HtStftFrameWindow();
extern "C" uint32_t HtStftStandalonePreload_HtStftSplit();
extern "C" uint32_t HtStftStandalonePreload_HtDftMatmul();
extern "C" uint32_t HtStftStandalonePreload_HtStftMergePack();

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cerrno>

using Clock=std::chrono::steady_clock;
static double Milliseconds(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static void Check(std::int64_t code,const char* operation) {
    if (!code) return;
    std::string message=std::string(operation)+" failed, code="+std::to_string(code);
    if (const char* detail=aclGetRecentErrMsg()) message+=std::string("\nACL detail: ")+detail;
    throw std::runtime_error(message);
}
#define CHECK_ACL(call) Check((call),#call)
constexpr std::size_t kWave=2*351232,kFrames=2*340*4096,kA=22*32*4096;
constexpr std::size_t kB=43*2*4096*48,kRaw=43*2*22*32*48,kCac=4*2048*336;

static std::vector<float> LoadFloat(const std::string& path,std::size_t count) {
    std::ifstream f(path.c_str(),std::ios::binary|std::ios::ate);
    if (!f || f.tellg()!=static_cast<std::streamoff>(count*sizeof(float))) throw std::runtime_error("Wrong FP32 input size: "+path);
    f.seekg(0);std::vector<float> x(count);
    if (!f.read(reinterpret_cast<char*>(x.data()),count*sizeof(float))) throw std::runtime_error("Cannot read "+path);
    for (float v:x) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite data: "+path);
    return x;
}
static void SaveFloat(const std::string& path,const std::vector<float>& x) {
    std::ofstream f(path.c_str(),std::ios::binary|std::ios::trunc);
    if (!f || !f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(float))) throw std::runtime_error("Cannot save "+path);
}
struct Buffer {void* ptr=nullptr;std::size_t bytes=0;std::string role;};
struct CopyCounts {std::size_t constants=0,input=0,output=0,constantBytes=0,inputBytes=0,outputBytes=0;};

struct Resources {
    int device=0;bool initialized=false,deviceSet=false,pending=false,inChain=false,outputReady=false;
    std::size_t completedChains=0,syncCount=0,copyOps=0;CopyCounts copies;
    aclrtContext context=nullptr;aclrtStream stream=nullptr;
    aclrtMemcpyKind toDevice=ACL_MEMCPY_HOST_TO_DEVICE,fromDevice=ACL_MEMCPY_DEVICE_TO_HOST;
    std::vector<Buffer> buffers;std::vector<aclTensor*> tensors;void* workspace=nullptr;uint64_t workspaceCapacity=0;
    ~Resources(){if(pending&&stream)(void)aclrtSynchronizeStream(stream);
        for(auto* t:tensors)if(t)(void)aclDestroyTensor(t);
        if(workspace)(void)aclrtFree(workspace);
        for(auto b:buffers)if(b.ptr)(void)aclrtFree(b.ptr);
        if(stream)(void)aclrtDestroyStream(stream);
        if(context)(void)aclrtDestroyContext(context);
        if(deviceSet)(void)aclrtResetDevice(device);
        if(initialized)(void)aclFinalize();}
    void Initialize(){CHECK_ACL(aclInit(nullptr));initialized=true;CHECK_ACL(aclrtSetDevice(device));deviceSet=true;
        CHECK_ACL(aclrtCreateContext(&context,device));CHECK_ACL(aclrtSetCurrentContext(context));CHECK_ACL(aclrtCreateStream(&stream));
        aclrtRunMode mode;CHECK_ACL(aclrtGetRunMode(&mode));if(mode!=ACL_HOST&&mode!=ACL_DEVICE)throw std::runtime_error("Invalid ACL mode");
        toDevice=mode==ACL_HOST?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
        fromDevice=mode==ACL_HOST?ACL_MEMCPY_DEVICE_TO_HOST:ACL_MEMCPY_DEVICE_TO_DEVICE;
        std::cout<<"run_mode="<<(mode==ACL_HOST?"ACL_HOST":"ACL_DEVICE")<<'\n';Preload();}
    void Preload();
    Buffer Allocate(const std::string& role,std::size_t bytes){Buffer b;b.bytes=bytes;b.role=role;
        CHECK_ACL(aclrtMalloc(&b.ptr,bytes,ACL_MEM_MALLOC_NORMAL_ONLY));buffers.push_back(b);return b;}
    aclTensor* View(const Buffer& b,const std::vector<int64_t>& dims,aclDataType dtype,std::size_t off=0){
        std::vector<int64_t> strides(dims.size(),1);std::size_t n=1;
        for(std::size_t i=dims.size();i>0;--i){if(dims[i-1]<=0)throw std::runtime_error("Invalid shape");strides[i-1]=n;n*=dims[i-1];}
        std::size_t item=dtype==ACL_FLOAT16?2:4;if(off>b.bytes/item||n>b.bytes/item-off||off*item%32)throw std::runtime_error("Invalid subview: "+b.role);
        auto* t=aclCreateTensor(dims.data(),dims.size(),dtype,strides.data(),0,ACL_FORMAT_ND,dims.data(),dims.size(),static_cast<uint8_t*>(b.ptr)+off*item);
        if(!t)throw std::runtime_error("Null tensor: "+b.role);
        tensors.push_back(t);return t;}
    void Sync(){CHECK_ACL(aclrtSynchronizeStream(stream));pending=false;++syncCount;}
    void* Workspace(uint64_t size){if(!size)return nullptr;if(size>workspaceCapacity){if(pending)Sync();if(workspace){CHECK_ACL(aclrtFree(workspace));workspace=nullptr;}
        CHECK_ACL(aclrtMalloc(&workspace,size,ACL_MEM_MALLOC_NORMAL_ONLY));workspaceCapacity=size;}return workspace;}
    void Upload(const Buffer& b,const void* p,std::size_t bytes){if(inChain||pending||bytes!=b.bytes)throw std::runtime_error("Upload inside chain/size mismatch");
        CHECK_ACL(aclrtMemcpy(b.ptr,b.bytes,p,bytes,toDevice));++copyOps;
        if(b.role=="waveform"){++copies.input;copies.inputBytes+=bytes;}
        else {++copies.constants;copies.constantBytes+=bytes;}}
    std::vector<float> Download(const Buffer& b){if(inChain||pending)throw std::runtime_error("Readback inside chain");
        std::vector<float> v(b.bytes/4);CHECK_ACL(aclrtMemcpy(v.data(),b.bytes,b.ptr,b.bytes,fromDevice));++copyOps;++copies.output;copies.outputBytes+=b.bytes;return v;}
};

struct Call {aclTensor *a=nullptr,*b=nullptr,*c=nullptr;};
struct RunResult {
    std::string tag;bool zero=false,warmup=false,pass=false;
    std::size_t kernels=0,syncs=0;
    double uploadMs=0,frameMs=0,splitMs=0,dftMs=0,mergeMs=0,pipelineMs=0;
    CopyCounts transfers;
};

class Pipeline {
public:
    Resources& r;
    Buffer wave,hann,frames,aHigh,aLow,bHigh,bLow,raw[4],cac;
    aclTensor *waveT=nullptr,*hannT=nullptr,*framesT=nullptr,*aHighT=nullptr,*aLowT=nullptr,*rawT[4]={},*cacT=nullptr;
    std::vector<Call> calls;
    explicit Pipeline(Resources& resources):r(resources) {}
    void Prepare(const std::vector<float>& window) {
        const auto start=Clock::now();
        wave=r.Allocate("waveform",kWave*sizeof(float));hann=r.Allocate("hann",4096*sizeof(float));
        frames=r.Allocate("frames",kFrames*sizeof(float));
        aHigh=r.Allocate("a_high",kA*sizeof(aclFloat16));aLow=r.Allocate("a_low",kA*sizeof(aclFloat16));
        bHigh=r.Allocate("b_high",kB*sizeof(aclFloat16));bLow=r.Allocate("b_low",kB*sizeof(aclFloat16));
        for (int i=0;i<4;++i) raw[i]=r.Allocate("raw_"+std::to_string(i),kRaw*sizeof(float));
        cac=r.Allocate("cac",kCac*sizeof(float));
        waveT=r.View(wave,{2,351232},ACL_FLOAT);hannT=r.View(hann,{4096},ACL_FLOAT);
        framesT=r.View(frames,{2,340,4096},ACL_FLOAT);
        aHighT=r.View(aHigh,{22,32,4096},ACL_FLOAT16);aLowT=r.View(aLow,{22,32,4096},ACL_FLOAT16);
        for (int i=0;i<4;++i) rawT[i]=r.View(raw[i],{43,2,22,32,48},ACL_FLOAT);
        cacT=r.View(cac,{1,4,2048,336},ACL_FLOAT);
        std::vector<aclFloat16> high(kB),low(kB);
        for (std::size_t block=0;block<43;++block) {
            const htdft::Band band={block*48,std::min<std::size_t>(48,2049-block*48)};
            for (std::size_t part=0;part<2;++part) {
                const auto plan=htdft::MakePlan(band,part==1);
                const std::size_t base=(block*2+part)*htdft::kBElements;
                for (std::size_t i=0;i<plan.size();++i) {
                    const float target=static_cast<float>(plan[i]);
                    high[base+i]=aclFloatToFloat16(target);
                    low[base+i]=aclFloatToFloat16(target-aclFloat16ToFloat(high[base+i]));
                }
            }
        }
        r.Upload(hann,window.data(),window.size()*sizeof(float));
        r.Upload(bHigh,high.data(),high.size()*sizeof(aclFloat16));
        r.Upload(bLow,low.data(),low.size()*sizeof(aclFloat16));
        aclTensor* av[2][22];aclTensor* bv[2][86];
        for (std::size_t b=0;b<22;++b) {
            av[0][b]=r.View(aHigh,{32,4096},ACL_FLOAT16,b*htdft::kAElements);
            av[1][b]=r.View(aLow,{32,4096},ACL_FLOAT16,b*htdft::kAElements);
        }
        for (std::size_t matrix=0;matrix<86;++matrix) {
            bv[0][matrix]=r.View(bHigh,{4096,48},ACL_FLOAT16,matrix*htdft::kBElements);
            bv[1][matrix]=r.View(bLow,{4096,48},ACL_FLOAT16,matrix*htdft::kBElements);
        }
        calls.reserve(7568);
        for (std::size_t matrix=0;matrix<86;++matrix) for (std::size_t batch=0;batch<22;++batch) {
            const std::size_t offset=(matrix*22+batch)*htdft::kCElements;
            for (int term=0;term<4;++term) {
                Call c;c.a=av[term/2][batch];c.b=bv[term%2][matrix];c.c=r.View(raw[term],{32,48},ACL_FLOAT,offset);
                calls.push_back(c);
            }
        }
        r.Workspace(2097152);
        std::size_t bytes=0;for (const auto& b:r.buffers) bytes+=b.bytes;
        std::cout<<"[DEVICE CACHE] constant_uploads="<<r.copies.constants
                 <<", constant_bytes="<<r.copies.constantBytes<<", tensor_views="<<r.tensors.size()
                 <<", allocated_buffer_bytes="<<bytes<<", preparation_wall_ms="<<Milliseconds(start)<<'\n';
    }
    RunResult RunDevice(const std::string& tag,const std::vector<float>& input,bool zero,bool warmup) {
        RunResult result;result.tag=tag;result.zero=zero;result.warmup=warmup;
        const CopyCounts before=r.copies;const auto beforeSync=r.syncCount;
        auto start=Clock::now();if(input.size()!=kWave)throw std::runtime_error("Wrong waveform input"); result.uploadMs=0;
        const auto pipelineStart=Clock::now();
        start=Clock::now();std::uint64_t size=0;aclOpExecutor* executor=nullptr;
        CHECK_ACL(aclnnHtStftFrameWindowGetWorkspaceSize(waveT,hannT,framesT,&size,&executor));
        if (!executor) throw std::runtime_error("Null frame executor");
        void* memory=r.Workspace(size);r.pending=true;
        CHECK_ACL(aclnnHtStftFrameWindow(memory,size,executor,r.stream));++result.kernels;r.Sync();result.frameMs=Milliseconds(start);
        start=Clock::now();size=0;executor=nullptr;
        CHECK_ACL(aclnnHtStftSplitGetWorkspaceSize(framesT,aHighT,aLowT,&size,&executor));
        if (!executor) throw std::runtime_error("Null split executor");
        memory=r.Workspace(size);r.pending=true;
        CHECK_ACL(aclnnHtStftSplit(memory,size,executor,r.stream));++result.kernels;r.Sync();result.splitMs=Milliseconds(start);
        start=Clock::now();std::size_t index=0;
        for (std::size_t block=0;block<43;++block) {
            for (std::size_t i=0;i<2*22*4;++i,++index) {
                const auto& call=calls[index];size=0;executor=nullptr;
                CHECK_ACL(aclnnHtDftMatmulGetWorkspaceSize(call.a,call.b,call.c,&size,&executor));
                if (!executor) throw std::runtime_error("Null MatMul executor");
                memory=r.Workspace(size);r.pending=true;
                CHECK_ACL(aclnnHtDftMatmul(memory,size,executor,r.stream));++result.kernels;
            }
            r.Sync();

        }
        result.dftMs=Milliseconds(start);
        start=Clock::now();size=0;executor=nullptr;
        CHECK_ACL(aclnnHtStftMergePackGetWorkspaceSize(rawT[0],rawT[1],rawT[2],rawT[3],cacT,&size,&executor));
        if (!executor) throw std::runtime_error("Null merge executor");
        memory=r.Workspace(size);r.pending=true;
        CHECK_ACL(aclnnHtStftMergePack(memory,size,executor,r.stream));++result.kernels;r.Sync();result.mergeMs=Milliseconds(start);
        r.outputReady=true;result.pipelineMs=Milliseconds(pipelineStart);
        result.syncs=r.syncCount-beforeSync;
        result.transfers.constants=r.copies.constants-before.constants;
        result.transfers.input=r.copies.input-before.input;result.transfers.output=r.copies.output-before.output;
        result.transfers.inputBytes=r.copies.inputBytes-before.inputBytes;result.transfers.outputBytes=r.copies.outputBytes-before.outputBytes;
        result.pass=result.kernels==7571 && result.syncs==46 && result.transfers.constants==0 &&
                    result.transfers.input==0 && result.transfers.output==0 && result.transfers.inputBytes==0;
        if (!result.pass) throw std::runtime_error("Unexpected frontend device/copy accounting");
        std::cout<<"[STFT DEVICE] tag="<<tag<<", pipeline_wall_ms="<<result.pipelineMs
                 <<", kernel_launches="<<result.kernels<<", syncs="<<result.syncs
                 <<", input_uploads=0, intermediate_host_copies=0, intermediate_readbacks=0\n";
        return result;
    }
};

inline void Resources::Preload(){
    std::cout<<"[PRELOAD] HtStftFrameWindow="<<HtStftStandalonePreload_HtStftFrameWindow()<<'\n';
    std::cout<<"[PRELOAD] HtStftSplit="<<HtStftStandalonePreload_HtStftSplit()<<'\n';
    std::cout<<"[PRELOAD] HtDftMatmul="<<HtStftStandalonePreload_HtDftMatmul()<<'\n';
    std::cout<<"[PRELOAD] HtStftMergePack="<<HtStftStandalonePreload_HtStftMergePack()<<'\n';
}
