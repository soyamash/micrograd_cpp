#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include "pool.h"

enum OP : int {
  kAdd = 1,
  kMul = 2,
  kPow = 3,
  kReLU = 4,
};

class Value {
 public:
  Value();
  Value(double d);
  double data_;
  double grad_;
  std::unordered_map<Value*, int> prev_;  // setとして使用するのでvalueはダミー
  OP op_;
  std::string label_;
  
  Value* add(Value* other);
  Value* neg();
  Value* sub(Value* other);
  Value* pow(Value* other);
  Value* div(Value* other);
  Value* mul(Value* other);
  Value* relu();

  void backward();
  void backwardElem();
  void update(double lr);
  
 private:
  Pool<Value> pool_;
  Value* other_ptr_;  // backward_op()のために保存
  Value* ret_ptr_;  // backward_op()のために保存
  int buildTopo(std::vector<Value*> &topo,
                 std::unordered_map<Value*, int> &visited, Value* v);
};

#endif /* ENGINE_H */
