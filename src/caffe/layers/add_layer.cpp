#include <cmath>
#include <vector>

#include "caffe/layers/add_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype> 
void AddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                  const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data_a = bottom[0]->cpu_data();
        const Dtype* bottom_data_b = bottom[1]->cpu_data();

        Dtype* top_data = top[0]->mutable_cpu_data();

        const int count = bottom[0]->count();
        caffe_add(count, bottom_data_a, bottom_data_b, top_data);
    }

INSTANTIATE_CLASS(AddLayer);
} // namespace caffe
