// Copyright 2024 IMAGINATION
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <uxl_common/uxl_common.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

using namespace std;

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>

// stl includes
#include <cstdlib>
#include <iostream>
#include <vector>
#include <ctime>

// SYCL Support
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// oneMKL Support
#include "oneapi/mkl.hpp"

namespace uxl_common
{

  class UxlFileLog
  {
    public:
      UxlFileLog(string strLogFile)
      : m_logFileName(strLogFile)
      {
        m_logFile.open (strLogFile);
        time_t timestamp;
        time(&timestamp);
        m_logFile << "UXL_COMMON: Log file created: " << ctime(&timestamp) << std::endl;
      }

      ~UxlFileLog()
      {
        m_logFile.close();
      }

    ofstream m_logFile;

    private:

      string m_logFileName;
  };

  int UxlCommonTestFunction()
  {
    UxlFileLog fUxlLog("uxl_common.log");
    
    // Creating SYCL queue
    sycl::queue Queue;

    // Report what the default device is
    fUxlLog.m_logFile << "UXL_COMMON: Using Compute Device - "
                      << Queue.get_device().get_info<sycl::info::device::name>() 
                      << std::endl;

    return 0;
  }

  template <typename fp>
  fp set_fp_value(fp arg1, fp /*arg2*/ = fp(0.0)) {
    return arg1;
  }

  template <typename fp>
  fp rand_scalar() {
      return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
  }

  template <typename vec>
  void rand_matrix(vec &M, oneapi::mkl::transpose trans, int m, int n, int ld) {
      using fp = typename vec::value_type;

      if (trans == oneapi::mkl::transpose::nontrans) {
          for (int j = 0; j < n; j++)
              for (int i = 0; i < m; i++)
                  M.at(i + j * ld) = rand_scalar<fp>();
      }
      else {
          for (int i = 0; i < m; i++)
              for (int j = 0; j < n; j++)
                  M.at(j + i * ld) = rand_scalar<fp>();
      }
  }

  int UxlCommonOneMKLTest()
  {
    UxlFileLog fUxlLog("uxl_common.log");

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception during GEMM:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    sycl::device dev = sycl::device();

    // Creating SYCL queue
    sycl::queue queue(dev, exception_handler);
    sycl::event gemm_done;

    // Report what the default device is
    fUxlLog.m_logFile << "UXL_COMMON: Starting oneMKL Test."
                      << queue.get_device().get_info<sycl::info::device::name>() 
                      << std::endl;

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

    // matrix data sizes
    int m = 45;
    int n = 98;
    int k = 67;

    // leading dimensions of data
    int ldA = 103;
    int ldB = 105;
    int ldC = 106;
    int sizea = (transA == oneapi::mkl::transpose::nontrans) ? ldA * k : ldA * m;
    int sizeb = (transB == oneapi::mkl::transpose::nontrans) ? ldB * n : ldB * k;
    int sizec = ldC * n;

    // set scalar fp values
    float alpha = set_fp_value(float(2.0), float(-0.5));
    float beta = set_fp_value(float(3.0), float(-1.5));


    // allocate matrix on host
    std::vector<float> A(sizea);
    std::vector<float> B(sizeb);
    std::vector<float> C(sizec);

    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C.begin(), C.end(), 0);

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, m, n, ldC);

    fUxlLog.m_logFile << "UXL_COMMON: Allocating shared memory."
                      << std::endl;

    // allocate memory on device
    auto dev_A = sycl::malloc_device<float>(sizea * sizeof(float), queue);
    auto dev_B = sycl::malloc_device<float>(sizeb * sizeof(float), queue);
    auto dev_C = sycl::malloc_device<float>(sizec * sizeof(float), queue);

    fUxlLog.m_logFile << "UXL_COMMON: Copy data from host to device."
                      << std::endl;

    // copy data from host to device
    queue.memcpy(dev_A, A.data(), sizea * sizeof(float)).wait();
    queue.memcpy(dev_B, B.data(), sizeb * sizeof(float)).wait();
    queue.memcpy(dev_C, C.data(), sizec * sizeof(float)).wait();

    fUxlLog.m_logFile << "UXL_COMMON: Executing GEMM function."
                      << std::endl;

    gemm_done = oneapi::mkl::blas::column_major::gemm(  queue, 
                                                        transA, 
                                                        transB, 
                                                        m, n, k, 
                                                        alpha,
                                                        dev_A, ldA, 
                                                        dev_B, ldB, 
                                                        beta, 
                                                        dev_C, ldC
                                                      );

    fUxlLog.m_logFile << "UXL_COMMON: Return from GEMM Function."
                      << std::endl;

    // Wait until calculations are done
    queue.wait_and_throw();

    fUxlLog.m_logFile << "UXL_COMMON: Wait and throw complete."
                      << std::endl;

    queue.memcpy(C.data(), dev_C, sizec * sizeof(float)).wait_and_throw();

    fUxlLog.m_logFile << "UXL_COMMON: Freeing device memory."
                      << std::endl;

    sycl::free(dev_C, queue);
    sycl::free(dev_B, queue);
    sycl::free(dev_A, queue);

    // Report what the default device is
    fUxlLog.m_logFile << "UXL_COMMON: Finishing oneMKL Test."
                      << std::endl;

    return 0;
  }

}  // namespace uxl_common
