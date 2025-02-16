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

int Neuron::init(size_t nin, bool nonlin) {
  nin_ = nin;
  nonlin_ = nonlin;
  std::default_random_engine gen(0);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (size_t i = 0; i < nin; i++) {
    weights_.emplace_back(pool_.allocate());
    weights_.back()->data_ = dis(gen);
  }
  bias_ = pool_.allocate();
  bias_->data_ = dis(gen);
  return 0;
}

Value* Neuron::operator()(std::vector<Value*> &x) {
  Value* act = pool_.allocate();
  act->data_ = 0.0;
  for (size_t x_idx = 0; x_idx < x.size(); x_idx++) {
    act = act->add(x[x_idx]->mul(weights_[x_idx]));
  }
  act = act->add(bias_);

  return nonlin_ ? act->relu() : act;
}

int Neuron::parameters(std::vector<Value*> &params) {
  params.reserve(weights_.size() + 1);
  for (Value* w : weights_) {
    params.emplace_back(w);
  }
  params.emplace_back(bias_);
  return 0;
}

int Neuron::showParameters() {
  std::cout << "weights: ";
  for (Value* w: this->weights_){
    std::cout << w->data_ << ", ";
  }
  std::cout << "bias: " << bias_->data_ <<std::endl;
  return 0;
}

int Layer::init(int nin, int nout) {
  nin_ = nin;
  nout_ = nout;
  total_params_ = (nin + 1) * nout;
  neurons_.reserve(nout);

  for (int i = 0; i < nout; i++){
    Neuron *n = pool_.allocate();
    n->init(nin, true);
    neurons_.emplace_back(n);
  }
  return 0;
}

int Layer::operator()(std::vector<Value *> &x, std::vector<Value* > &ret){
  ret.clear();
  ret.reserve(neurons_.size());
  for (Neuron *n: neurons_){
    ret.emplace_back((*n)(x));
  }
  return 0;
}

int Layer::parameters(std::vector<Value*> &params) {
  params.reserve(total_params_);

  for (Neuron *n: neurons_) {
    std::vector<Value*> n_params;
    n->parameters(n_params);
    for(Value* w: n_params){
      params.emplace_back(w);
    }
  }
  return 0;
}

int Layer::showParameters() {
  std::cout << "Layer Weights: " << total_params_ << std::endl;
  for (Neuron *n : neurons_) {
    n->showParameters();
  }
  return 0;
}

int MLP::init(int nin, std::vector<int> &nout) {
  nin_ = nin;
  nout_ = nout.back();
  layers_.reserve(nout.size() + 1);

  total_params_=0;
  for (int i = 0; i < nout.size(); i++){
    if (i == 0) {
      Layer *l = pool_.allocate();
      l->init(nin, nout[0]);
      layers_.push_back(l);
      total_params_ += nin * nout[0];
    } else {
      Layer *l = pool_.allocate();
      l->init(nout[i - 1], nout[i]);
      layers_.push_back(l);
      total_params_ += nout[i - 1] * nout[i];
    }
  }
  return 0;
}

int MLP::operator()(std::vector<Value *> &x, std::vector<Value *> &ret){
  for (auto l: layers_){
    (*l)(x, ret);
    x = ret;
  }
  return 0;
}

int MLP::parameters(std::vector<Value*> &params) {
  params.reserve(total_params_);
  for (Layer *l : layers_) {
    std::vector<Value*> l_params;
    l->parameters(l_params);
    for (Value* w : l_params){
      params.emplace_back(w);
    }
  }
  return 0;
}

int MLP::showParameters() {
  for (size_t i = 0; i < layers_.size(); i++) {
    std::cout << "Layer" << i<< ": " << std::endl;
    layers_[i]->showParameters();
  }
  return 0;
}



