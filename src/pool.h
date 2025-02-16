#ifndef POOL_H
#define POOL_H

#include <vector>

template<typename T>
class Pool {
 public:
  Pool() {};
  ~Pool() {
    clear();
  };

  T* get() {
    if (pool_ptr_ % kDefaultSize == 0) {
      pool_.push_back(new T[kDefaultSize]);
    }
    T* ret = &pool_[pool_ptr_ / kDefaultSize][pool_ptr_ % kDefaultSize];
    pool_ptr_++;
    return ret;
  };

  int clear() {
    for (T* e : pool_) {
      delete[] e;
    }
    pool_.clear();
    pool_ptr_ = 0;
    return 0;
  }

 private:
  const size_t kDefaultSize = 1024;
  std::vector<T*> pool_;
  size_t pool_ptr_ = 0;
};


#endif /* POOL_H */
