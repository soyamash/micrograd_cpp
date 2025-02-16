#include <iostream>
#include <fstream>
#include <cstdint>
#include <unordered_map>

#include "engine.h"


Value::Value(): data_(0.0), grad_(0.0), other_ptr_(nullptr), ret_ptr_(nullptr) {};
Value::Value(double d): data_(d), grad_(0.0), other_ptr_(nullptr), ret_ptr_(nullptr) {};

Value* Value::add(Value* other) {
  Value* ret = pool_.get();
  ret->data_ = data_ + other->data_;
  ret->prev_[other] = 0;  // setとして使用するのでvalueはダミー
  ret->prev_[this] = 0;  // setとして使用するのでvalueはダミー
  
  op_ = OP::kAdd;
  other_ptr_ = other; // backward_op()のために保存
  ret_ptr_ = ret; // backward_op()のために保存
  return ret;
}
Value* Value::neg() {
  Value* ret = pool_.get();
  ret->data_ = -1.0;
  return this->mul(ret);
}
Value* Value::sub(Value* other) {
  return this->add(other->neg());
}
Value* Value::pow(Value* other) {
  Value* ret = pool_.get();
  ret->data_ = std::pow(data_, other->data_);
  ret->prev_[other] = 0;  // setとして使用するのでvalueはダミー
  ret->prev_[this] = 0;  // setとして使用するのでvalueはダミー
  
  op_ = OP::kPow;
  other_ptr_ = other; // backward_op()のために保存
  ret_ptr_ = ret; // backward_op()のために保存
  return ret;
}
Value* Value::div(Value* other) {
  Value* ret = pool_.get();
  ret->data_ = -1.0;
  return this->mul(other->pow(ret));
}
Value* Value::mul(Value* other) {
  Value* ret = pool_.get();
  ret->data_ = data_ * other->data_;
  ret->prev_[other] = 0;  // setとして使用するのでvalueはダミー
  ret->prev_[this] = 0;  // setとして使用するのでvalueはダミー
  
  op_ = OP::kMul;
  other_ptr_ = other; // backward_op()のために保存
  ret_ptr_ = ret; // backward_op()のために保存
  return ret;
}

Value* Value::relu() {
  Value* ret = pool_.get();
  ret->data_ = (data_ > 0.0) ? data_ : 0.0;
  ret->prev_[this] = 0;  // setとして使用するのでvalueはダミー
  
  op_ = OP::kReLU;
  ret_ptr_ = ret; // backward_op()のために保存
  return ret;
}

void Value::backward() {
  std::vector<Value*> topo;
  std::unordered_map<Value*, int> visited;  // setとして使用するのでvalueはダミー

  buildTopo(topo, visited, this);

  grad_ = 1.0;

  for (auto itr = topo.rbegin(); itr != topo.rend(); ++itr) {
    (*itr)->backwardElem();
  }
}

int Value::buildTopo(std::vector<Value*> &topo,
                      std::unordered_map<Value*, int> &visited, Value* v) {
  if (visited.find(v) == visited.end()) {
    visited[v] = 0;

    for (auto itr = v->prev_.begin(); itr != v->prev_.end(); ++itr) {
      buildTopo(topo, visited, itr->first);
    }
    topo.push_back(v);
  }
  return 0;
}

void Value::backwardElem() {
  switch (op_) {
  case OP::kAdd: {
    if (ret_ptr_ != nullptr && other_ptr_ != nullptr) {
      grad_ += ret_ptr_->grad_;
      other_ptr_->grad_ += ret_ptr_->grad_;  
    }
    break;
  }
  case OP::kMul: {
    if (ret_ptr_ != nullptr && other_ptr_ != nullptr) {
      grad_ += other_ptr_->data_ * ret_ptr_->grad_;
      other_ptr_->grad_ += data_ * ret_ptr_->grad_;
    }
    break;
  }
  case OP::kPow: {
    if (ret_ptr_ != nullptr && other_ptr_ != nullptr) {
      grad_ += other_ptr_->data_ * std::pow(data_, other_ptr_->data_ - 1) * ret_ptr_->grad_;
    }
    break;
  }
  case OP::kReLU: {
    if (ret_ptr_ != nullptr) {
      grad_ += (ret_ptr_->data_ > 0) * ret_ptr_->grad_;
    }
    break;
  }
  default: {
    break;
  }
  }
  return;
}

void Value::update(double lr) {
  data_ -= lr * grad_;
  return;
}
