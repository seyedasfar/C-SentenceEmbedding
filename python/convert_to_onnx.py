import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class SentenceTransformerWrapper(nn.Module):
    def __init__(self, model):
        super(SentenceTransformerWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model({'input_ids': input_ids, 'attention_mask': attention_mask})['sentence_embedding']

def convert_model_to_onnx():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.eval()

    # Move model to CPU
    model.to('cpu')

    # Wrap the model
    wrapped_model = SentenceTransformerWrapper(model)

    # Get the tokenizer from the model
    tokenizer = model.tokenizer

    # Create dummy input
    dummy_input = tokenizer(["This is a dummy input"], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    # Move dummy input to CPU
    dummy_input = {key: value.to('cpu') for key, value in dummy_input.items()}

    # Define input and output names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["sentence_embedding"]

    # Export the model to ONNX
    torch.onnx.export(
        wrapped_model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        "sentence_transformer.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"}
        },
        opset_version=14
    )

    print("Model has been converted to ONNX and saved as 'sentence_transformer.onnx'")

if __name__ == "__main__":
    convert_model_to_onnx()
