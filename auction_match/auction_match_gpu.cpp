#include <torch/torch.h>

void AuctionMatchLauncher(int b,int n,const float * xyz1,const float * xyz2,int * matchl,int * matchr,float * cost);

int auction_match_wrapper_fast(int b, int n, 
    at::Tensor xyz1_tensor, at::Tensor xyz2_tensor, at::Tensor matchl_tensor, 
    at::Tensor matchr_tensor, at::Tensor cost_tensor) {

    const float *xyz1 = xyz1_tensor.data<float>();
    const float *xyz2 = xyz2_tensor.data<float>();
    int *matchl = matchl_tensor.data<int>();
    int *matchr = matchr_tensor.data<int>();
    float *cost = cost_tensor.data<float>();

    AuctionMatchLauncher(b, n, xyz1, xyz2, matchl, matchr, cost);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("auction_match_cuda", &auction_match_wrapper_fast, "auction_match_wrapper_fast forward");
}