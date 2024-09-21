#!/bin/bash

# Navigate to the cprog directory
cd "$(dirname "$0")"

# Compile the C program
gcc -o onnx_inference main.c -I/opt/homebrew/Cellar/onnxruntime/1.17.1/include/onnxruntime -L/opt/homebrew/Cellar/onnxruntime/1.17.1/lib -lonnxruntime

# List of sentences to test
sentences=(
    "This is a test sentence."
    "Let's see how fast we can generate embeddings."
    "Performance testing is crucial for optimization."
)

# Measure the total time taken
start_time=$(date +%s.%N)

for sentence in "${sentences[@]}"; do
    ./onnx_inference "$sentence"
done

end_time=$(date +%s.%N)
time_taken=$(echo "$end_time - $start_time" | bc)

echo "Total time taken for embedding ${#sentences[@]} sentences: $time_taken seconds"
