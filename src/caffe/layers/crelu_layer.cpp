#include <algorithm>
#include <vector>

#include "caffe/layers/crelu_layer.hpp"

namespace caffe {
template <typename Dtype>
void CReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
        CHECK_NE(bottom[0], top[0]) << "CReLU doesn't support inplace computation!" 
            << " Please check prototxt!";
        negative_slope = this->layer_param_.crelu_param().negative_slope();
        concat_axis_ = bottom[0]->
            CanonicalAxisIndex(this->layer_param_.crelu_param().concat_axis());
        CHECK_LT(concat_axis_, bottom[0]->num_axes()) << "concat axis out of range.";
}

template <typename Dtype> 
void CReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape = bottom[0].shape();
    top_shape[concat_axis_] = 2 * bottom[0]->shape(concat_axis_);
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
inline Dtype ReLU(Dtype x, Dtype negative_slope) {
    return std::max(x, Dtype(0)) + negative_slope * std::min(x, Dtype(0));
}

template <typename Dtype> 
void CReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
    consr vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count(concat_axis_);
    for (int i = 0; i < bottom[0]->count(0, concat_axis_); ++i) {
        top_data[2*i*count + j] = ReLU(bottom_data[i * count + j], negative_slope_);
        top_data[(2*i+1)*count + j] = ReLU(-bottom_data[i * count + j], negative_slope_);
    }
}

INSTANTIATE_CLASS(CReLULayer);
} // namespace caffe
