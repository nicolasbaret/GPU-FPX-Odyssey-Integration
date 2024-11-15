#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 \"expression\""
    echo "Example: $0 \"x * x\""
    exit 1
fi

# Escape the expression for the C preprocessor
EXPRESSION=$1
ESCAPED_EXPRESSION=$(echo "$EXPRESSION" | sed 's/[()]/\\&/g')

OUTPUT_NAME="cuda_program"

echo "Compiling with expression: $EXPRESSION"
nvcc dynamic_cuda.cu -o $OUTPUT_NAME "-DDEVICE_FUNCTION_BODY=$ESCAPED_EXPRESSION"

if [ $? -eq 0 ]; then
    echo "Compilation successful. Run with: ./$OUTPUT_NAME"
else
    echo "Compilation failed"
fi