#include "stft_pipeline.h"
#include <sstream>
static void Require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static std::string Json(const std::string& s){std::ostringstream f;f<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')f<<'\\'<<c;else if(c=='\n')f<<"\\n";else if(c<32)f<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else f<<c;}f<<'"';return f.str();}
struct CaseTiming {std::string name;bool warmup=false;double reset=0,input=0,read=0,wall=0;RunResult compute;};
int main(int argc,char** argv){
    try {
        Require(argc==4,"Usage: stft_test DEVICE DATA_DIR RESULT_DIR");int device=std::stoi(argv[1]);Require(device>=0,"Invalid device");
        const std::string data=argv[2],out=argv[3];std::cout<<std::setprecision(12);
        const std::vector<std::string> names={"tones_warmup","impulse","tones","noise","zero","tones_restore"};
        const auto setup=Clock::now();Resources r;r.device=device;r.Initialize();Pipeline pipeline(r);
        pipeline.Prepare(LoadFloat(data+"/window_scaled_f32.bin",4096));const double setupMs=Milliseconds(setup);
        std::vector<float> sentinel(kCac,std::numeric_limits<float>::quiet_NaN());std::vector<CaseTiming> records;
        for(std::size_t i=0;i<names.size();++i){CaseTiming c;c.name=names[i];c.warmup=i==0;
            const auto x=LoadFloat(data+"/"+c.name+"_input_f32.bin",kWave);
            // Test initialization is outside the live device chain and its timing.
            auto start=Clock::now();CHECK_ACL(aclrtMemcpy(pipeline.cac.ptr,pipeline.cac.bytes,sentinel.data(),pipeline.cac.bytes,r.toDevice));++r.copyOps;c.reset=Milliseconds(start);
            const auto wall=Clock::now();start=Clock::now();r.Upload(pipeline.wave,x.data(),pipeline.wave.bytes);c.input=Milliseconds(start);
            const auto before=r.copyOps;r.inChain=true;c.compute=pipeline.RunDevice(c.name,x,c.name=="zero",c.warmup);
            r.inChain=false;Require(!r.pending&&r.copyOps==before,"Application copy inside live STFT chain");++r.completedChains;
            start=Clock::now();const auto y=r.Download(pipeline.cac);c.read=Milliseconds(start);c.wall=Milliseconds(wall);
            std::size_t bad=0;for(float v:y)bad+=!std::isfinite(v);Require(bad==0,"Unwritten sentinel/nonfinite output: "+c.name);
            SaveFloat(out+"/"+c.name+"_npu_f32.bin",y);records.push_back(c);
            std::cout<<"[CASE COMPLETE] "<<c.name<<", wall_ms="<<c.wall<<", frame_ms="<<c.compute.frameMs<<", split_ms="<<c.compute.splitMs<<", dft_ms="<<c.compute.dftMs<<", merge_ms="<<c.compute.mergeMs<<'\n';
        }
        std::ofstream f((out+"/run_report.json").c_str());Require(bool(f),"Cannot write run report");
        f<<std::setprecision(12)<<"{\"status\":\"RUN_COMPLETE_REQUIRES_NUMERICAL_VALIDATION\",\"device\":"<<device<<",\"setup_ms\":"<<setupMs<<",\"constant_uploads\":"<<r.copies.constants<<",\"constant_bytes\":"<<r.copies.constantBytes<<",\"workspace_bytes\":"<<r.workspaceCapacity<<",\"runs\":[";
        for(std::size_t i=0;i<records.size();++i){if(i)f<<',';const auto& c=records[i];const auto& t=c.compute;
            f<<"{\"name\":"<<Json(c.name)<<",\"warmup\":"<<(c.warmup?"true":"false")<<",\"sentinel_reset_ms\":"<<c.reset<<",\"input_transfer_ms\":"<<c.input<<",\"frame_ms\":"<<t.frameMs<<",\"split_ms\":"<<t.splitMs<<",\"dft_ms\":"<<t.dftMs<<",\"merge_ms\":"<<t.mergeMs<<",\"device_chain_ms\":"<<t.pipelineMs<<",\"output_transfer_ms\":"<<c.read<<",\"segment_wall_ms\":"<<c.wall<<",\"kernel_launches\":"<<t.kernels<<",\"syncs\":"<<t.syncs<<",\"intermediate_application_copies\":0,\"input_bytes\":"<<pipeline.wave.bytes<<",\"output_bytes\":"<<pipeline.cac.bytes<<'}';
        }
        f<<"]}\n";f.close();Require(bool(f),"Cannot close run report");std::cout<<"COMPLETE: six cases, reused constants/device buffers. Run validate.py for numerical PASS.\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
