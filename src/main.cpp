#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "pool.h"
#include "engine.h"
#include "nn.h"


int main () {
  Pool<Value> pool;
  Value* value1 = pool.get();
  value1->data_ = 2.5; value1->label_ = "1";
  Value* value2 = pool.get();
  value2->data_ = 3.7; value2->label_ = "2";
  Value* value3 = pool.get();
  value3->data_ = -3.0; value3->label_ = "3";
  Value* value4 = pool.get();
  value4->data_ = 1.7; value4->label_ = "4";

  Value* value_1_2 = value1->add(value2);
  value_1_2->label_ = "1_2";
  Value* value_1_2_3 = value_1_2->mul(value3);
  value_1_2_3->label_ = "1_2_3";
  Value* result_final = value_1_2_3->add(value4);
  result_final->label_ = "result_final";


  // Access the result and print the data
  std::cout << "Result Final: " << result_final->data_ << std::endl;
  // result_final->set_grad(1.0);
  result_final->backward();

  std::cout<<value1->label_<<":"<<value1->data_<<" grad: "<<value1->grad_<<std::endl;
  std::cout<<value2->label_<<":"<<value2->data_<<" grad: "<<value2->grad_<<std::endl;
  std::cout<<value3->label_<<":"<<value3->data_<<" grad: "<<value3->grad_<<std::endl;
  std::cout<<value4->label_<<":"<<value4->data_<<" grad: "<<value4->grad_<<std::endl;
  
  return 0;
}