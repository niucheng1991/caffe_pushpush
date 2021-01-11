#ifndef CAFFE_ADD_LAYER_HPP_
#define CAFFE_ADD_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
    
template <typename Dtype>
class AddLayer : public NeuronLayer<Dtype> {
public:
    explicit AddLayer(const LayerParameter& param) : NeuronLayer<Dtype> (param) {}
    
    virtual inline const char* type() const { return "Add"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                              const vector<bool>& propagate_down, 
                              const vector<Blob<Dtype>*>& bottom) {
        NOT_IMPLEMENTED;
    }    
};

}

#endif // CAFFE_ADD_LAYER_HPP_