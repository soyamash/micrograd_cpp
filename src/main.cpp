#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "pool.h"
#include "engine.h"
#include "nn.h"


int main () {
  Pool<Value> pool;
  Value* value1 = pool.allocate();
  value1->data_ = 2.5; value1->label_ = "1";
  Value* value2 = pool.allocate();
  value2->data_ = 3.7; value2->label_ = "2";

  std::vector<int> layer_dims;
  layer_dims.push_back(2);
  layer_dims.push_back(3);
  MLP m;
  m.init(2, layer_dims);
  
  std::vector<Value*> input;
  input.push_back(value1);
  input.push_back(value2);

  for (int i = 0; i < 10; i++) {
    std::vector<Value*> ret;
    m(input, ret);
    for (Value *r : ret) {
      std::cout << r->data_ << ", ";
    }
    std::cout << std::endl;
    Value *loss = pool.allocate();
    for (Value *r : ret) {
      Value* target = pool.allocate();
      target->data_ = 1.0;
      r = r->sub(target);
      Value* twice = pool.allocate();
      twice->data_ = 2.0;
      r = r->pow(twice);
      loss = loss->add(r);
    }
    m.zeroGrad();
    loss->backward();
    loss->update(0.01);
  }
  return 0;
}