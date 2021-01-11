#ifndef CAFFE_LAYERS_CRELU_LAYER_HPP_
#define CAFFE_LAYERS_CRELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

/**
* CReLU(x) = [ ReLU(x), ReLU(-x) ]
*/
namespace caffe {
template <typename Dtype> 
class CReLULayer : public NeuronLayer<Dtype> {
public:
    explicit CReLULayer(const LayerParameter& param) :
        NeuronLayer<Dtype>(param) {}
    virtual inline const char* type() const { return "CReLU"; }
protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) {
            NOT_IMPLEMENTED;
    }

    int concat_axis;
    Dtype negative_slope_;
};


}

#endif // CAFFE_LAYERS_CRELU_LAYER_H_
