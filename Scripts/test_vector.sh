#!/bin/bash
#
# Copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This script tests whether muRISCV-NN V vector code passes the unit tests

# Prevent silent failures
set -euo pipefail

source config.sh

################################################################################
############## Download or install required software ###########################
################################################################################

# Install LLVM 14 (which includes vector support)
if clang-14 --version &>/dev/null; then
  echo "LLVM 14 appears to be installed."
else
  echo "No LLVM 14 installation found. Installing LLVM 14..."
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh
  sudo ./llvm.sh 14
  rm llvm.sh
fi

# Download rv32gcv GCC
if [ -d ${TC_DIR_RV32GCV} ]; then
  echo "Found rv32gcv GCC compiler in the Toolchain directory."
else
  echo "No rv32gcv GCC compiler in the Toolchain directory found. Downloading one..."
  (
    cd ${TC_DIR}
    ./download_rv32gcv.sh
  )
fi

# Download rv32imv GCC
if [ -d ${TC_DIR_RV32IMV} ]; then
  echo "Found rv32imv GCC compiler in the Toolchain directory."
else
  echo "No rv32imv GCC compiler in the Toolchain directory found. Downloading one..."
  (
    cd ${TC_DIR}
    ./download_rv32imv.sh
  )
fi

# Download OVPsim
if [ -f ${SIM_BIN_PATH_OVP} ]; then
  echo "Found an OVPsim instance in the Sim directory."
else
  echo "No OVPsim in the Sim directory found. Downloading one..."
  (
    cd ${SIM_BIN_DIR_OVP}
    ./download.sh
  )
fi

# Download Spike
if [ -f ${SIM_BIN_PATH_SPIKE} ]; then
  echo "Found an Spike instance in the Sim directory."
else
  echo "No Spike in the Sim directory found. Downloading one..."
  (
    cd ${SIM_BIN_DIR_SPIKE}
    ./download.sh
  )
  # Install Spike dependencies
  sudo apt-get install libboost-all-dev
  sudo apt-get install device-tree-compiler
fi

# Loop over all common VLENs
for vl in "${VLENS[@]}"; do
  ################################################################################
  #################### Test on OVPsim rv32gcv ####################################
  ################################################################################

  # Configure and build with LLVM
  rm -rf ${BUILD_DIR}
  mkdir ${BUILD_DIR}
  cmake -B ${BUILD_DIR} -S .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSIMULATOR=OVPsim -DTOOLCHAIN=LLVM -DRISCV_GCC_PREFIX=${TC_DIR_RV32GCV} -DVLEN=${vl} -DUSE_VEXT=ON -DUSE_PEXT=OFF
  make -j $(nproc) -C ${BUILD_DIR}
  (
    cd ${BUILD_DIR}
    ctest --verbose
  )

  # Configure and build with GCC
  rm -rf ${BUILD_DIR}
  mkdir ${BUILD_DIR}
  cmake -B ${BUILD_DIR} -S .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSIMULATOR=OVPsim -DTOOLCHAIN=GCC -DRISCV_GCC_PREFIX=${TC_DIR_RV32GCV} -DVLEN=${vl} -DUSE_VEXT=ON -DUSE_PEXT=OFF
  make -j $(nproc) -C ${BUILD_DIR}
  (
    cd ${BUILD_DIR}
    ctest --verbose
  )

  ################################################################################
  #################### Test on Spike rv32gcv #####################################
  ################################################################################

  # Configure and build with LLVM
  rm -rf ${BUILD_DIR}
  mkdir ${BUILD_DIR}
  cmake -B ${BUILD_DIR} -S .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSIMULATOR=Spike -DTOOLCHAIN=LLVM -DRISCV_GCC_PREFIX=${TC_DIR_RV32GCV} -DVLEN=${vl} -DUSE_VEXT=ON -DUSE_PEXT=OFF
  make -j $(nproc) -C ${BUILD_DIR}
  (
    cd ${BUILD_DIR}
    ctest --verbose
  )

  # Configure and build with GCC
  rm -rf ${BUILD_DIR}
  mkdir ${BUILD_DIR}
  cmake -B ${BUILD_DIR} -S .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSIMULATOR=Spike -DTOOLCHAIN=GCC -DRISCV_GCC_PREFIX=${TC_DIR_RV32GCV} -DVLEN=${vl} -DUSE_VEXT=ON -DUSE_PEXT=OFF
  make -j $(nproc) -C ${BUILD_DIR}
  (
    cd ${BUILD_DIR}
    ctest --verbose
  )

  ################################################################################
  #################### Test on Spike rv32imv #####################################
  ################################################################################

  # Configure and build with LLVM
  rm -rf ${BUILD_DIR}
  mkdir ${BUILD_DIR}
  cmake -B ${BUILD_DIR} -S .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSIMULATOR=Spike -DTOOLCHAIN=LLVM -DRISCV_GCC_PREFIX=${TC_DIR_RV32IMV} -DVLEN=${vl} -DUSE_VEXT=ON -DUSE_PEXT=OFF -DRISCV_ARCH=rv32imzve32x -DRISCV_ABI=ilp32
  make -j $(nproc) -C ${BUILD_DIR}
  (
    cd ${BUILD_DIR}
    ctest --verbose
  )

  # Configure and build with GCC
  rm -rf ${BUILD_DIR}
  mkdir ${BUILD_DIR}
  cmake -B ${BUILD_DIR} -S .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSIMULATOR=Spike -DTOOLCHAIN=GCC -DRISCV_GCC_PREFIX=${TC_DIR_RV32IMV} -DVLEN=${vl} -DUSE_VEXT=ON -DUSE_PEXT=OFF -DRISCV_ARCH=rv32imv -DRISCV_ABI=ilp32
  make -j $(nproc) -C ${BUILD_DIR}
  (
    cd ${BUILD_DIR}
    ctest --verbose
  )

done
