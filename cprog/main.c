#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>

// Define a simplified vocabulary (this is a basic example; in practice, use a complete vocabulary)
const char* vocabulary[] = {
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "python", "javascript", "is", "a", "language"
};

// Function to find the index of a word in the vocabulary
int find_vocab_index(const char* word) {
    for (int i = 0; i < sizeof(vocabulary) / sizeof(vocabulary[0]); i++) {
        if (strcmp(vocabulary[i], word) == 0) {
            return i;
        }
    }
    return 1; // [UNK] token index
}

// Helper function to check ONNX Runtime status
void check_status(OrtStatus* status, const OrtApi* api) {
    if (status != NULL) {
        const char* msg = api->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        api->ReleaseStatus(status);
        exit(1);
    }
}

// Function to tokenize input text (simplified example)
void tokenize(const char* text, int64_t* input_ids, int64_t* attention_mask, int max_length) {
    char* token = strtok(strdup(text), " ");
    int i = 0;

    // Add [CLS] token at the beginning
    input_ids[i] = 2; // [CLS] token index
    attention_mask[i] = 1;
    i++;

    while (token != NULL && i < max_length - 1) {
        input_ids[i] = find_vocab_index(token);
        attention_mask[i] = 1;
        token = strtok(NULL, " ");
        i++;
    }

    // Add [SEP] token at the end
    input_ids[i] = 3; // [SEP] token index
    attention_mask[i] = 1;
    i++;

    // Pad the rest with [PAD] tokens
    for (; i < max_length; i++) {
        input_ids[i] = 0; // [PAD] token index
        attention_mask[i] = 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <sentence>\n", argv[0]);
        return 1;
    }

    const char* sentence = argv[1];

    // Get the ONNX Runtime API
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // Initialize ONNX Runtime
    OrtEnv* env;
    check_status(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env), api);

    // Create the session options
    OrtSessionOptions* session_options;
    check_status(api->CreateSessionOptions(&session_options), api);
    check_status(api->SetIntraOpNumThreads(session_options, 1), api);
    check_status(api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC), api);

    // Load the ONNX model
    const char* model_path = "../python/sentence_transformer.onnx"; // Adjust path according to your structure
    OrtSession* session;
    check_status(api->CreateSession(env, model_path, session_options, &session), api);

    // Create input tensors
    OrtMemoryInfo* memory_info;
    check_status(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), api);

    // Input data arrays
    int64_t input_ids[1][128] = {{0}};
    int64_t attention_mask[1][128] = {{0}};

    // Tokenize input text
    tokenize(sentence, input_ids[0], attention_mask[0], 128);

    // Create input tensor for input_ids
    int64_t input_ids_shape[2] = {1, 128};
    OrtValue* input_ids_tensor = NULL;
    check_status(api->CreateTensorWithDataAsOrtValue(memory_info, input_ids, sizeof(input_ids),
                                                     input_ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_ids_tensor), api);
    
    // Create input tensor for attention_mask
    int64_t attention_mask_shape[2] = {1, 128};
    OrtValue* attention_mask_tensor = NULL;
    check_status(api->CreateTensorWithDataAsOrtValue(memory_info, attention_mask, sizeof(attention_mask),
                                                     attention_mask_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &attention_mask_tensor), api);

    // Prepare input and output names
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"sentence_embedding"};

    // Run the model
    OrtValue* output_tensor = NULL;
    const OrtValue* input_tensors[] = {input_ids_tensor, attention_mask_tensor};
    check_status(api->Run(session, NULL, input_names, input_tensors, 2, output_names, 1, &output_tensor), api);

    // Get output tensor values
    float* output_tensor_values;
    check_status(api->GetTensorMutableData(output_tensor, (void**)&output_tensor_values), api);

    // Print the output in a format compatible with the shell script
    printf("Sentence: %s\n", sentence);
    printf("Embedding: [");
    for (int i = 0; i < 5; i++) { // Print first 5 dimensions for brevity
        if (i > 0) printf(" ");
        printf("%f", output_tensor_values[i]);
    }
    printf("]...\n");

    // Clean up
    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_ids_tensor);
    api->ReleaseValue(attention_mask_tensor);
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(session_options);
    api->ReleaseEnv(env);

    return 0;
}
