#include <random>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <unordered_map>

#include "nn.h"

Module::~Module() {}

int Module::zeroGrad() {
  std::vector<Value*> params;
  parameters(params);
  for (Value* w: params) {
    w->grad_ = 0.0;
  }
  return 0;
}
 
int Module::update(double lr) {
  std::vector<Value*> params;
  parameters(params);
  for (Value* w: params) {
    w->update(lr);
  }
  return 0; 
}

Neuron::Neuron(size_t nin, bool nonlin): nonlin_(nonlin) {
  std::default_random_engine gen(0);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (size_t i = 0; i < nin; i++) {
    weight_.emplace_back(pool_.get());
    weight_.back()->data_ = dis(gen);
  }
  bias_ = pool_.get();
  bias_->data_ = dis(gen);
}

Value* Neuron::operator()(std::vector<Value*> &x) {
  Value* act = pool_.get();
  act->data_ = 0.0;
  for (size_t x_idx = 0; x_idx < x.size(); x_idx++) {
    act = act->add(x[x_idx]->mul(weight_[x_idx]));
  }
  act = act->add(bias_);

  return nonlin_ ? act->relu() : act;
}

int Neuron::parameters(std::vector<Value*> &params) {
  params.reserve(weight_.size() + 1);

  for (Value* w : weight_) {
    params.emplace_back(w);
  }
  params.emplace_back(bias_);
  return 0;
}

int Neuron::showParameters() {
  std::cout << "weights: ";
  for (Value* w: this->weight_){
    std::cout << w->data_ << ", ";
  }
  std::cout << "bias: " << bias_->data_ <<std::endl;
  return 0;
}

Layer::Layer(int nin, int nout) {
  total_params_ = (nin + 1) * nout;
  neurons_.reserve(nout + 1);

  for (int i = 0; i < nout; i++){
    Neuron n(nin, true);
    neurons_.emplace_back(n);
  }
}

int Layer::operator()(std::vector<Value* > &ret, std::vector<Value*> &x){
  ret.reserve(neurons_.size() + 1);
  for (Neuron &n: neurons_){
    ret.emplace_back(n(x));
  }
  return 0;
}

int Layer::parameters(std::vector<Value*> &params) {
  params.reserve(total_params_);

  for (Neuron &n: neurons_) {
    std::vector<Value*> n_params;
    n.parameters(n_params);
    for(Value* w: n_params){
      params.emplace_back(w);
    }
  }
  return 0;
}

int Layer::showParameters() {
  std::cout << "Layer Weights: " << total_params_ << std::endl;
  for (Neuron &n : neurons_) {
    n.showParameters();
  }
  return 0;
}


