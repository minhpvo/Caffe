#ifdef USE_MPI

#include "caffe/common.hpp"
#include "caffe/util/mpi_job_queue.hpp"
#include "caffe/util/mpi_templates.hpp"


using boost::shared_ptr;
using boost::mutex;
using boost::lock_guard;

namespace caffe {

template <typename Dtype>
shared_ptr<MPIJobQueue<Dtype> > MPIJobQueue<Dtype>::singleton_;

template <typename Dtype>
MPIJobQueue<Dtype>::MPIJobQueue()
  : is_running_(false),
    thread_started_(false) {
  try {
#ifndef CPU_ONLY
    if (cudaSuccess != cudaGetDevice(&device_id_)) {
      device_id_ = -1;
    }
#endif
    is_running_.store(true);
    thread_.reset(new boost::thread(&MPIJobQueue<Dtype>::ThreadFunc, this));
  } catch (...) {
    LOG(FATAL) << "Failed to start MPI job queue thread.";
  }
}

template <typename Dtype>
MPIJobQueue<Dtype>::~MPIJobQueue<Dtype>() {
  try {
    is_running_.store(false);
    cv_work_.notify_one();
    thread_->join();
  } catch (...) {
    LOG(FATAL) << "Failed to release MPI job queue thread.";
  }
}

template <typename Dtype>
void MPIJobQueue<Dtype>::ThreadFunc() {
#ifndef CPU_ONLY
  if (device_id_ >= 0) {
    CUDA_CHECK(cudaSetDevice(device_id_));
  }
#endif
  thread_started_.store(true);
  while (true) {
    mutex::scoped_lock read_lock(queue_mutex_);
    while (queue_.empty() && is_running_.load()) {
      cv_work_.wait(read_lock);
    }
    read_lock.unlock();

    if (!is_running_.load()) break;

    Dispatch(queue_.front());
    mutex::scoped_lock write_lock(queue_mutex_);
    queue_.pop();
    write_lock.unlock();
    cv_done_.notify_one();
  }
  while (!queue_.empty()) {
    lock_guard<mutex> lk(queue_mutex_);
    Dispatch(queue_.front());
    queue_.pop();
    cv_done_.notify_one();
  }
}

template <typename Dtype>
void MPIJobQueue<Dtype>::WaitAll() {
  mutex::scoped_lock lk(queue_mutex_);
  while (!queue_.empty()) {
    cv_done_.wait(lk);
  }
}

template <typename Dtype>
void MPIJobQueue<Dtype>::Push(const MPIJobQueue<Dtype>::Job& job) {
  while (!thread_started_.load());
  mutex::scoped_lock push_lock(queue_mutex_);
  queue_.push(job);
  push_lock.unlock();
  cv_work_.notify_one();
}

template <typename Dtype>
void MPIJobQueue<Dtype>::Dispatch(MPIJobQueue<Dtype>::Job& job) {
  switch (job.op) {
    case OP_SUM_ALL:
      MPIAllreduce<Dtype>(job.count,
          (job.src_ptr == job.dst_ptr) ? MPI_IN_PLACE : job.src_ptr,
          job.dst_ptr, MPI_SUM);
      break;
    case OP_ALL_GATHER:
      MPIAllgather<Dtype>(job.count, job.src_ptr, job.dst_ptr);
      break;
    case OP_SCATTER:
      MPIScatter<Dtype>(job.count, job.src_ptr, job.dst_ptr);
      break;
    case OP_BCAST:
      MPIBcast<Dtype>(job.count, job.src_ptr);
      break;
    default:
      LOG(FATAL) << "Unrecognized MPI job";
  }
}

INSTANTIATE_CLASS(MPIJobQueue);

}

#endif