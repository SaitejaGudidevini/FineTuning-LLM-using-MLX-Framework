# YouTube Comment Reply Generator

This project implements an AI-powered YouTube comment reply generation system using DialoGPT fine-tuned on YouTube comments and replies data.

## Overview

The system uses Microsoft's DialoGPT model fine-tuned on a dataset of YouTube comments and their corresponding replies. It can automatically generate contextually relevant and appropriate responses to user comments on YouTube videos.

## Features

- Fine-tuned DialoGPT model for YouTube comment reply generation
- Support for Apple Silicon via MPS (Metal Performance Shaders)
- Custom dataset handling for comment-reply pairs
- Configurable generation parameters (temperature, top-k, top-p)

## Requirements

- Python 3.8+
- PyTorch with MPS support
- Transformers library
- Pandas
- MLX (optional for additional acceleration)

## Dataset

The system uses two datasets:
1. A sample YouTube comments dataset with predefined replies
2. A custom dataset (`comments_data.csv`) containing:
   - Original commenter information
   - Original comments
   - Channel replies

## Usage

### Training

The model is fine-tuned using a custom dataset class that handles the comment-reply pairs. The training process involves:

1. Loading and preprocessing the data
2. Creating a custom dataset and dataloader
3. Fine-tuning the DialoGPT model
4. Saving the fine-tuned model for inference

### Inference

To generate replies for new comments:

```python
def generate_reply(comment):
    inputs = tokenizer(comment + tokenizer.eos_token, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    reply_ids = model.generate(
        inputs=input_ids, 
        attention_mask=attention_mask, 
        max_length=100, 
        pad_token_id=tokenizer.eos_token_id, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.9, 
        do_sample=True
    )

    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply
```

## Model Parameters

- **Temperature**: Controls randomness (0.7 default)
- **Top-k**: Limits vocabulary to top k tokens (50 default)
- **Top-p**: Nucleus sampling parameter (0.9 default)
- **Max length**: Maximum token length for generated replies (100 default)

## Future Improvements

- Implement sentiment analysis to match reply tone with comment sentiment
- Add more diverse training data
- Implement content moderation for inappropriate comments/replies
- Create a web interface for easy testing

## License

This project is available for educational and research purposes.
