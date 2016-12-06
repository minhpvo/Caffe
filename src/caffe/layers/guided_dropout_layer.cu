#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void MakeMaskDeterministic(const int n,
    const Dtype* guidance, const Dtype guidance_threshold,
    Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    mask[index] = (guidance[index] > guidance_threshold);
  }
}

template <typename Dtype>
__global__ void ComputeScaleDeterministic(const int num, const int channels,
    const Dtype* mask, Dtype* scale) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype sum = 0;
    for (int i = 0; i < channels; ++i) {
      sum += mask[index * channels + i];
    }
    scale[index] = sum;
  }
}

template <typename Dtype>
__global__ void MakeMaskStochastic(const int n,
    const Dtype* guidance, const Dtype sigmoid_scaler,
    Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    mask[index] = (mask[index] * (1. + exp(-guidance[index] * sigmoid_scaler)) <= 1);
  }
}

template <typename Dtype>
__global__ void ComputeScaleStochastic(const int num, const int channels,
    const Dtype* guidance, const Dtype sigmoid_scaler, Dtype* scale) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype sum = 0;
    for (int i = 0; i < channels; ++i) {
      sum += 1. / (1. + exp(-guidance[index * channels + i] * sigmoid_scaler));
    }
    scale[index] = sum;
  }
}

template <typename Dtype>
__global__ void DropoutForward(
    const int count, const int dim, const int channels, const int siz,
    const Dtype* in, const Dtype* mask, const Dtype* scale, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = in[index] * mask[index / dim];
    if (scale[index / siz] > 0) {
      out[index] = out[index] * channels / scale[index / siz];
    }
  }
}

template <typename Dtype>
void GuidedDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* guidance = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* mask = rand_vec_.mutable_gpu_data();
  Dtype* scale = scale_.mutable_gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int dim = count / num / channels;
  const int siz = count / num;
  if (type_ == "deterministic") {
    MakeMaskDeterministic<Dtype><<<CAFFE_GET_BLOCKS(num * channels),
      CAFFE_CUDA_NUM_THREADS>>>(
        num * channels, guidance, guidance_threshold_, mask);
    ComputeScaleDeterministic<Dtype><<<CAFFE_GET_BLOCKS(num),
      CAFFE_CUDA_NUM_THREADS>>>(
        num, channels, mask, scale);
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, dim, channels, siz,
        bottom_data, mask, scale, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    if (this->phase_ == TRAIN) {
      caffe_gpu_rng_uniform<Dtype>(num * channels, 0, 1, mask);
      MakeMaskStochastic<<<CAFFE_GET_BLOCKS(num * channels),
        CAFFE_CUDA_NUM_THREADS>>>(
          num * channels, guidance, sigmoid_scaler_, mask);
      ComputeScaleStochastic<<<CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS>>>(
          num, channels, guidance, sigmoid_scaler_, scale);
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, dim, channels, siz,
          bottom_data, mask, scale, top_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(count, bottom_data, top_data);
    }
  }
}

template <typename Dtype>
__global__ void DropoutBackward(
    const int count, const int channels, const int dim, const int siz,
    const Dtype* in_diff, const Dtype* mask, const Dtype* scale, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, count) {
    out_diff[index] = in_diff[index] * mask[index / dim];
    if (scale[index / siz] > 0) {
      out_diff[index] = out_diff[index] * siz / scale[index / siz];
    }
  }
}

template <typename Dtype>
void GuidedDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* mask = rand_vec_.gpu_data();
    const Dtype* scale = scale_.gpu_data();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int dim = count / num / channels;
    const int siz = count / num;
    if (type_ == "deterministic") {
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, dim, siz, top_diff, mask, scale, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      if (this->phase_ == TRAIN) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
            count, channels, dim, siz, top_diff, mask, scale, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        caffe_copy(count, top_diff, bottom_diff);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GuidedDropoutLayer);


}  // namespace caffe
