#!/usr/bin/env bash
set -e
set -o pipefail

#================================================================
#
# Prepare.
#
#================================================================

sourceDir=$1
resultDir=$2
generatorName=$3

if [[ $# -ne 3 ]] ; then
    echo 'Arguments are required: sourceDir resultDir generatorName'
    exit 1
fi

#================================================================
#
# Make GPU compiler.
#
#================================================================

echo "Building GPU compiler..."

mkdir -p "$resultDir"
cd "$resultDir"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$resultDir \
    -G"$generatorName" "$sourceDir" \
    -DHEXLIB_PLATFORM=0 \
    -DHEXLIB_GUARDED_MEMORY=0

cmake --build . --target gpuCompiler --
