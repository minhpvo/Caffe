// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void GuidedDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  type_ = this->layer_param_.guided_dropout_param().type();
  if (type_ != "deterministic" && type_ != "stochastic") {
    LOG(FATAL) << "The type of guided dropout layers should be either "
                  "\"deterministic\" or \"stochastic\"";
  }
  guidance_threshold_ = this->layer_param_.guided_dropout_param()
                                          .guidance_threshold();
  sigmoid_scaler_ = this->layer_param_.guided_dropout_param().sigmoid_scaler();
}

template <typename Dtype>
void GuidedDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  scale_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void GuidedDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* guidance = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* mask = rand_vec_.mutable_cpu_data();
  Dtype* scale = scale_.mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int dim = count / num / channels;
  const int siz = count / num;
  if (type_ == "deterministic") {
    for (int n = 0; n < num; ++n) {
      scale[n] = 0;
      for (int c = 0; c < channels; ++c) {
        const int index = n * channels + c;
        if (guidance[index] > guidance_threshold_) {
          mask[index] = 1;
          ++scale[n];
        } else {
          mask[index] = 0;
        }
      }
    }
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i / dim];
      if (scale[i / siz] > 0) {
        top_data[i] = top_data[i] * channels / scale[i / siz];
      }
    }
  } else {
    if (this->phase_ == TRAIN) {
      caffe_rng_uniform<Dtype>(count, 0, 1, mask);
      for (int n = 0; n < num; ++n) {
        scale[n] = 0;
        for (int c = 0; c < channels; ++c) {
          const int index = n * channels + c;
          const Dtype reserve_prob = sigmoid(guidance[index] * sigmoid_scaler_);
          mask[index] = (mask[index] <= reserve_prob);
          scale[n] += reserve_prob;
        }
      }
      for (int i = 0; i < count; ++i) {
        top_data[i] = bottom_data[i] * mask[i / dim];
        if (scale[i / siz] > 0) {
          top_data[i] = top_data[i] * channels / scale[i / siz];
        }
      }
    } else {
      caffe_copy(count, bottom_data, top_data);
    }
  }
}

template <typename Dtype>
void GuidedDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* mask = rand_vec_.cpu_data();
    const Dtype* scale = scale_.cpu_data();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int dim = count / num / channels;
    const int siz = count / num;
    if (type_ == "deterministic") {
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i / dim];
        if (scale[i / siz] > 0) {
          bottom_diff[i] = bottom_diff[i] * channels / scale[i / siz];
        }
      }
    } else {
      if (this->phase_ == TRAIN) {
        for (int i = 0; i < count; ++i) {
          bottom_diff[i] = top_diff[i] * mask[i / dim];
          if (scale[i / siz] > 0) {
            bottom_diff[i] = bottom_diff[i] * channels / scale[i / siz];
          }
        }
      } else {
        caffe_copy(count, top_diff, bottom_diff);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GuidedDropoutLayer);
#endif

INSTANTIATE_CLASS(GuidedDropoutLayer);
REGISTER_LAYER_CLASS(GuidedDropout);

}  // namespace caffe
