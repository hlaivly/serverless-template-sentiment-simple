import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLMAutoModelForSeq2SeqLM


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    # global model
    # global tokenizer
    global classifier
    device = 0 if torch.cuda.is_available() else -1

    # Flan-T5 version, if changed be sure to update in download.py too
    # model_name = "knkarthick/MEETING_SUMMARY"
    classifier = pipeline("sentiment-analysis", model="sbcBI/sentiment_analysis")
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    # global model
    global classifier

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # output = model.generate(input_ids, max_length=100)
    # result = tokenizer.decode(output[0], skip_special_tokens=True)
    result = classifier(prompt)[0]["label"]

    # Return the results as a dictionary
    return result