#!/bin/bash


# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 \"expression\""
    echo "Example: $0 \"pow(e, sin((pow((x + 1.0), 2.0) - 3.0))) / log(x)\""
    exit 1
fi

OUTPUT_NAME="cuda_program"

#  Create a temporary file with .cu extension
# mktemp creates a unique temporary file to avoid conflicts
# --suffix=.cu is needed so nvcc recognizes it as CUDA source
TMP_FILE=$(mktemp --suffix=.cu)

# Write to the temporary file:
# 1. First, define the macro DEVICE_FUNCTION_BODY as the provided expression
# 2. Then include all the content from the original CUDA file
cat > "$TMP_FILE" << EOL
#define DEVICE_FUNCTION_BODY ($1)
$(cat driver-program.cu)
EOL

# Show what we're compiling
echo "Compiling with expression: $1"

# Compile using nvcc:
# - "$TMP_FILE" is the source file we just created
# - -o $OUTPUT_NAME specifies the output executable name
# - -arch=native tells nvcc to compile for the current GPU
nvcc "$TMP_FILE" -o  $OUTPUT_NAME --expt-relaxed-constexpr

# Save the compilation result (0 = success, non-zero = failure)
RESULT=$?

# # Clean up by removing the temporary file
rm "$TMP_FILE"

# Report success or failure
if [ $RESULT -eq 0 ]; then
    echo "Compilation successful. Run with: ./$OUTPUT_NAME"
else
    echo "Compilation failed"
fi