# uxl_common

## Purpose

This package contains a library of common functions related to UXL oneMKL.

## Information

You will need to export the following values to allow the uxl_common module to build with SYCL:

export DPCPP_HOME=~/sycl_workspace
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export CPATH=$DPCPP_HOME/llvm/build/include:$CPATH
export CPATH=$DPCPP_HOME/llvm/build/include/sycl:$CPATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/install/lib:$LD_LIBRARY_PATH

## This is the one that makes autoware work finding sycl library:
export LIBRARY_PATH=$DPCPP_HOME/llvm/build/install/lib:$LIBRARY_PATH
