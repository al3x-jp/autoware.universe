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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
//#include "oneapi/mkl.hpp"

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
    fUxlLog.m_logFile << "INFO: Using Compute Device - "
                      << Queue.get_device().get_info<sycl::info::device::name>() 
                      << std::endl;

    return 0;
  }

}  // namespace uxl_common
