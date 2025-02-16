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
};

class Neuron: public Module {
 public:
  Neuron() {};
  ~Neuron() {};
  int init(size_t nin, bool nonlin = true);
  Value* operator()(std::vector<Value*> &x);
  int parameters(std::vector<Value*> &params);
  int showParameters();
 private:
  int nin_;
  std::vector<Value *> weights_;
  Value* bias_;
  bool nonlin_;
  Pool<Value> pool_;
};

class Layer: public Module{
 public:
  Layer() {};
  ~Layer() {};
  int init(int nin, int nout);
  int operator()(std::vector<Value *> &x, std::vector<Value* > &ret);
  int parameters(std::vector<Value*> &params);
  int showParameters();
 private:
  int nin_;
  int nout_;
  std::vector<Neuron *> neurons_;
  int total_params_;
  Pool<Neuron> pool_;
};

class MLP: public Module{
 public:
  MLP() {};
  ~MLP() {};
  int init(int nin, std::vector<int> &nout);
  int operator()(std::vector<Value *> &x, std::vector<Value* > &ret);
  int parameters(std::vector<Value*> &params);
  int showParameters();
 private:
  int nin_;
  int nout_;
  std::vector<Layer *> layers_;
  int total_params_;
  Pool<Layer> pool_;
};

#endif /* NN_H */
