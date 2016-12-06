#ifdef USE_MPI
#ifndef MPI_JOB_QUEUE_HPP_
#define MPI_JOB_QUEUE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <queue>

namespace caffe {

template <typename Dtype>
class MPIJobQueue {
public:
  enum OperationType {
    OP_SUM_ALL,
    OP_ALL_GATHER,
    OP_SCATTER,
    OP_BCAST
  };

  class Job {
  public:
    Job(OperationType op_, int count_, Dtype* src_ptr_, Dtype* dst_ptr_)
      : op(op_), count(count_), src_ptr(src_ptr_), dst_ptr(dst_ptr_) {}
    OperationType op;
    int count;
    Dtype* src_ptr;
    Dtype* dst_ptr;
  };

  ~MPIJobQueue();
  inline static MPIJobQueue<Dtype>& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new MPIJobQueue<Dtype>());
    }
    return *singleton_;
  }

  inline static void Synchronize() { Get().WaitAll(); }
  inline static void PushSumAll(const int count, Dtype* data) {
    Get().Push(MPIJobQueue<Dtype>::Job(OP_SUM_ALL, count, data, data));
  }
  inline static void PushAllgather(const int count,
                                   const Dtype* from, Dtype* to) {
    Get().Push(MPIJobQueue<Dtype>::Job(OP_ALL_GATHER, count,
                                            const_cast<Dtype*>(from), to));
  }
  inline static void PushScatter(const int count,
                                 const Dtype* from, Dtype* to) {
    Get().Push(MPIJobQueue<Dtype>::Job(OP_SCATTER, count,
                                            const_cast<Dtype*>(from), to));
  }
  inline static void PushBcast(const int count, Dtype* data) {
    Get().Push(MPIJobQueue<Dtype>::Job(OP_BCAST, count, data, data));
  }

private:
  MPIJobQueue();

  void ThreadFunc();
  void WaitAll();
  void Push(const MPIJobQueue<Dtype>::Job& job);
  void Dispatch(MPIJobQueue<Dtype>::Job& job);

  int device_id_;

  std::queue<MPIJobQueue<Dtype>::Job> queue_;
  mutable boost::mutex queue_mutex_;
  boost::atomic<bool> is_running_;
  boost::atomic<bool> thread_started_;
  boost::shared_ptr<boost::thread> thread_;
  boost::condition_variable cv_work_;
  boost::condition_variable cv_done_;

  static shared_ptr<MPIJobQueue<Dtype> > singleton_;

  DISABLE_COPY_AND_ASSIGN(MPIJobQueue);
};

}

#endif // MPI_JOB_QUEUE_HPP_
#endif // USE_MPI