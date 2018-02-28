/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <numeric>
#include <string>
#include "mkldnn.hpp"

using namespace mkldnn;
using namespace std;

extern "C"
{

void Convolution( const int batch,
                  const int nk,
                  const int nimage1,
                  const int nimage2,
                  const int n1, const int n2, 
                  float* net_src,       // Y (batch, n2, nimage1, nimage2)
                  float* conv_weights,  // K (n1, n2, nk, nk)
                  float* net_dst )   //      (batch, n1, nimage1, nimage2)  Output
  {                   
    auto cpu_engine = engine(engine::cpu, 0);

    const auto srcformat     = memory::format::nchw;
    const auto weightsformat = memory::format::oihw;
  //  const auto srcformat     = memory::format::nhwc;
  //  const auto weightsformat = memory::format::hwio;

    //std::vector<float> net_src(batch * 3 * 227 * 227);
    //std::vector<float> net_dst(batch * 96 * 27 * 27);

    /* AlexNet: conv
     * {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 227, 227}
     * strides: {1, 1}
     */
    memory::dims conv_src_tz = {batch, n2, nimage1, nimage2};
  //  memory::dims conv_src_tz = {batch, nimage1, nimage2, n2};
    memory::dims conv_weights_tz = {n1, n2, nk, nk};
  //  memory::dims conv_weights_tz = {nk, nk, n1, n2};
    memory::dims conv_bias_tz = {n1};
    memory::dims conv_dst_tz = {batch, n1, nimage1, nimage2};
 //   memory::dims conv_dst_tz = {batch, nimage1, nimage2, n1};
    memory::dims conv_strides = {1, 1};
    auto conv_padding = {1, 1};

    vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
        conv_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv_user_src_memory = memory({{{conv_src_tz}, memory::data_type::f32,
        srcformat}, cpu_engine}, net_src);
    auto conv_user_weights_memory = memory({{{conv_weights_tz},
        memory::data_type::f32, weightsformat}, cpu_engine},
        conv_weights);
    auto conv_user_bias_memory = memory({{{conv_bias_tz},
        memory::data_type::f32, memory::format::x}, cpu_engine},
        conv_bias.data());

    /* create memory descriptors for convolution data w/ no specified format */
    auto conv_src_md = memory::desc({conv_src_tz}, memory::data_type::f32,
        memory::format::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, memory::data_type::f32,
        memory::format::any);
    auto conv_weights_md = memory::desc({conv_weights_tz},
        memory::data_type::f32, memory::format::any);


    auto conv_user_dst_memory = memory({{{conv_dst_tz}, memory::data_type::f32,
        srcformat}, cpu_engine}, net_dst );

    auto conv_dst_md = memory::desc({conv_dst_tz}, memory::data_type::f32,
        memory::format::any);

    

    /* create a convolution */
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md, conv_strides, conv_padding, conv_padding,
        padding_kind::zero);
    auto conv_prim_desc =
        convolution_forward::primitive_desc(conv_desc, cpu_engine);


    auto conv_dst_memory = conv_user_dst_memory;
    if (memory::primitive_desc(conv_prim_desc.dst_primitive_desc()) !=
        conv_user_dst_memory.get_primitive_desc()) {
        conv_dst_memory = memory(conv_prim_desc.dst_primitive_desc());
    }


    std::vector<primitive> net;

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto conv_src_memory = conv_user_src_memory;
    if (memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
        conv_user_src_memory.get_primitive_desc()) {
        conv_src_memory = memory(conv_prim_desc.src_primitive_desc());
        net.push_back(reorder(conv_user_src_memory, conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
        conv_user_weights_memory.get_primitive_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_primitive_desc());
        net.push_back(reorder(conv_user_weights_memory, conv_weights_memory));
    }




    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, conv_src_memory,
        conv_weights_memory, conv_user_bias_memory, conv_dst_memory));


    if (conv_dst_memory != conv_user_dst_memory) {
        net.push_back(reorder(conv_dst_memory, conv_user_dst_memory));
    }



    stream(stream::kind::eager).submit(net).wait();
}  // Convolution


}  // extern "C"


// int main(int argc, char **argv) {
//     try {

//         const int batch = 8, nK=3, nimage=227, n1=96, n2=3;

//         vector<float> net_src(batch * n2 * nimage * nimage);
//         vector<float> conv_weights(n1 * n2 * nK * nK);
//         vector<float> net_dst(batch * n1 * nimage * nimage);

//         Convolution( batch, nK, nimage,nimage, n1, n2, net_src.data(), conv_weights.data(), net_dst.data() );
//     }
//     catch(error& e) {
//         std::cerr << "status: " << e.status << std::endl;
//         std::cerr << "message: " << e.message << std::endl;
//     }
//     return 0;
// }
