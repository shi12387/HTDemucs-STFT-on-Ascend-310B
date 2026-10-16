#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace htdft {
constexpr std::size_t kRows=32,kFft=4096,kColumns=48;
constexpr std::size_t kAElements=kRows*kFft,kBElements=kFft*kColumns,kCElements=kRows*kColumns;
struct Band {std::size_t start,count;};
inline std::vector<double> MakePlan(Band band,bool imaginary){
    if(band.count>48 || band.start+band.count>2049)throw std::runtime_error("Invalid one-sided DFT band");
    std::vector<double> b(kBElements,0.0);const double pi=std::acos(-1.0);
    for(std::size_t n=0;n<kFft;++n)for(std::size_t j=0;j<band.count;++j){
        const std::size_t k=band.start+j;const double phase=2.0*pi*((n*k)%kFft)/kFft;
        b[n*kColumns+j]=imaginary?-std::sin(phase):std::cos(phase);
    }
    return b;
}
}
