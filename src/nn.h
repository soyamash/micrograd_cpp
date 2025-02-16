#ifndef NN_H
#define NN_H

#include <string>
#include <vector>

#include "pool.h"
#include "engine.h"

class Module {
 public:
  virtual ~Module() = 0;
  int zeroGrad();
  virtual int parameters(std::vector<Value*> &params) = 0;
  int update(double lr);
  Pool<Value> pool_;
};

class Neuron: public Module {
 public:
  Neuron(size_t nin, bool nonlin = true);
  Value* operator()(std::vector<Value*> &x);
  int parameters(std::vector<Value*> &params);
  int showParameters();
 private:
  std::vector<Value*> weight_;
  Value* bias_;
  bool nonlin_;
};

class Layer: public Module{
 public:
  Layer(int nin, int nout);
  int operator()(std::vector<Value* > &ret, std::vector<Value *> &x);
  int parameters(std::vector<Value*> &params);
  int showParameters();
 private:
  std::vector<Neuron> neurons_;
  int total_params_;
};

#endif /* NN_H */
