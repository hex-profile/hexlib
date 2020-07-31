#!/usr/bin/env bash
set -e
set -o pipefail

#================================================================
#
# Prepare.
#
#================================================================

sourceDir=$1
intermDir=$2
resultDir=$3
generatorName=$4

if [[ $# -ne 4 ]] ; then
    echo 'Arguments are required: sourceDir intermDir resultDir generatorName'
    exit 1
fi

#================================================================
#
# Make GPU compiler.
#
#================================================================

echo "Building GPU compiler..."

mkdir -p "$resultDir"
mkdir -p "$intermDir"
cd "$intermDir"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$resultDir \
    -G"$generatorName" "$sourceDir" \
    -DHEXLIB_PLATFORM=0 \
    -DHEXLIB_GUARDED_MEMORY=0

cmake --build . --target gpuCompiler --
