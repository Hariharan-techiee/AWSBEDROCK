# import boto3
# import json
# prompt_data="""write a poem about struggle of IT industry"""


# bedrock=boto3.client(service_name="bedrock-runtime")

# payload={
    
#     "prompt":"[INST]"+ prompt_data + "[/INST]",
#     "max_gen_len":50,
#     "temperature":0.5,
#     "top_p":0.9
# }

# body=json.dumps(payload)

# model_id="Llama 3 8B Instruct"
# response=bedrock.invoke_model(
#     body=body,
#     modelId=model_id,
#     accept="application/json",
#     contentType="application/json"
    
    
# )

# response_body=json.loads(response.get("body".read()))
# response_text=response_body['generation']
# print(response_text)
import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock')

# List available models (optional)
available_models = bedrock.list_models()
print("Available Models:", available_models)

# Invoke the model (replace with a valid modelId)
response = bedrock.invoke_model(
    modelId='valid-model-identifier',
    contentType='text/plain',
    body='Your input text here'
)

# Process the response
print(response['body'].read().decode('utf-8'))
