#include <algorithm>
#include <vector>

#include "caffe/layers/my_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MySoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer__ ( new SoftmaxLayer<Dtype>(this->layer_param_) );
  softmax_layer_ = softmax_layer__;
  // LayerParameter softmax_param;
  // softmax_param.set_type("Softmax");
  // softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);

  top[0]->ReshapeLike(*bottom[0]);
  softmax_bottom_vec_.clear();
  nClassLabel = this->layer_param_.my_softmax_param().stride();
  nRotation = bottom[0]->count() / (bottom[0]->num() * nClassLabel );
  // NOTICE bottom[0] should be reshaped!
  Blob<Dtype> bottomR( bottom[0]->num() * nRotation, nClassLabel, bottom[0]->height(), bottom[0]->width() );
  bottomR.set_cpu_data( (Dtype*)bottom[0]->cpu_data() );
  softmax_bottom_vec_.push_back(&bottomR);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void MySoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> bottomR( bottom[0]->num() * nRotation, nClassLabel, bottom[0]->height(), bottom[0]->width() );
  bottomR.set_cpu_data( (Dtype*)bottom[0]->cpu_data() );
  softmax_bottom_vec_[0] = &bottomR;
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();

  Dtype* top_data = top[0]->mutable_cpu_data();
  memcpy(top_data, prob_data, sizeof(Dtype) * bottom[0]->count());
}

template <typename Dtype>
void MySoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // NOT IMPLEMENTED
}


#ifdef CPU_ONLY
STUB_GPU(MySoftmaxLayer);
#endif

INSTANTIATE_CLASS(MySoftmaxLayer);

}  // namespace caffe
