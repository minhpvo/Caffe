#ifdef USE_MPI
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mpi_job_queue.hpp"

namespace caffe {

template <typename Dtype>
void MPIScatterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size())
      << "The number of bottom and top blobs must be the same";
}

template <typename Dtype>
void MPIScatterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    vector<int> shape = bottom[i]->shape();
    if (Caffe::mpi_size() > 1) {
      shape[0] /= Caffe::mpi_size();
    }
    top[i]->Reshape(shape);
  }
}

template <typename Dtype>
void MPIScatterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (Caffe::mpi_size() > 1) {
    for (int i = 0; i < bottom.size(); ++i) {
      MPIJobQueue<Dtype>::PushScatter(bottom[i]->count(), bottom[i]->cpu_data(),
          top[i]->mutable_cpu_data());
    }
    MPIJobQueue<Dtype>::Synchronize();
  } else {
    for (int i = 0; i < bottom.size(); ++i) {
      top[i]->ShareData(*bottom[i]);
    }
  }
}

template <typename Dtype>
void MPIScatterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (Caffe::mpi_size() > 1) {
    for (int i = 0; i < bottom.size(); ++i) {
      MPIJobQueue<Dtype>::PushAllgather(bottom[i]->count(), top[i]->cpu_diff(),
          bottom[i]->mutable_cpu_diff());
    }
    MPIJobQueue<Dtype>::Synchronize();
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_scal(bottom[i]->count(), Dtype(1. / Caffe::mpi_size()),
                 bottom[i]->mutable_cpu_diff());
    }
  } else {
    for (int i = 0; i < bottom.size(); ++i) {
      bottom[i]->ShareDiff(*top[i]);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MPIScatterLayer);
#endif

INSTANTIATE_CLASS(MPIScatterLayer);
REGISTER_LAYER_CLASS(MPIScatter);

} // namespace caffe

#endif // USE_MPI