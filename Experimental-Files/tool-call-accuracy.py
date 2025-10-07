'''
Check what is the format of data required for evaluation. 
'''
import os
from azure.ai.evaluation import ToolCallAccuracyEvaluator, AzureOpenAIModelConfiguration
from pprint import pprint

openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
)


tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)

query = "How is the weather in Seattle ?"
query = "Can you send me an email with weather information for Seattle?"
response = [
    {
        "createdAt": "2025-03-26T17:27:35Z",
        "run_id": "run_zblZyGCNyx6aOYTadmaqM4QN",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_CUdbkBfvVBla2YP3p24uhElJ",
                "name": "fetch_weather",
                "arguments": {"location": "Seattle"},
            }
        ],
    },
    {
        "createdAt": "2025-03-26T17:27:37Z",
        "run_id": "run_zblZyGCNyx6aOYTadmaqM4QN",
        "tool_call_id": "call_CUdbkBfvVBla2YP3p24uhElJ",
        "role": "tool",
        "content": [{"type": "tool_result", "tool_result": {"weather": "Rainy, 14\u00b0C"}}],
    },
    {
        "createdAt": "2025-03-26T17:27:38Z",
        "run_id": "run_zblZyGCNyx6aOYTadmaqM4QN",
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_iq9RuPxqzykebvACgX8pqRW2",
                "name": "send_email",
                "arguments": {
                    "recipient": "your_email@example.com",
                    "subject": "Weather Information for Seattle",
                    "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
                },
            }
        ],
    },
    {
        "createdAt": "2025-03-26T17:27:41Z",
        "run_id": "run_zblZyGCNyx6aOYTadmaqM4QN",
        "tool_call_id": "call_iq9RuPxqzykebvACgX8pqRW2",
        "role": "tool",
        "content": [
            {"type": "tool_result", "tool_result": {"message": "Email successfully sent to your_email@example.com."}}
        ],
    },
    {
        "createdAt": "2025-03-26T17:27:42Z",
        "run_id": "run_zblZyGCNyx6aOYTadmaqM4QN",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I have successfully sent you an email with the weather information for Seattle. The current weather is rainy with a temperature of 14\u00b0C.",
            }
        ],
    },
]

tool_definitions = [
    {
        "name": "fetch_weather",
        "description": "Fetches the weather information for the specified location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "The location to fetch weather for."}},
        },
    },
    {
        "name": "send_email",
        "description": "Sends an email with the specified subject and body to the recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Email address of the recipient."},
                "subject": {"type": "string", "description": "Subject of the email."},
                "body": {"type": "string", "description": "Body content of the email."},
            },
        },
    },
]

response = tool_call_accuracy(query=query, response=response, tool_definitions=tool_definitions)
pprint(response)
       
from azure.ai.evaluation import evaluate

# This sample files contains the evaluation data in JSONL format. Where each line is a run from agent.
# This was saved using agent thread and converter.
file_name = "evaluation_data.jsonl"

response = evaluate(
    data=file_name,
    evaluation_name="Tool Call Accuracy Evaluation",
    evaluators={
        "tool_call_accuracy": tool_call_accuracy,
    },
    azure_ai_project={
        "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
        "project_name": "ayushija-dummy-resource",
        "resource_group_name": "rg-ayushija-2422",
    },
)
pprint(f'AI Foundary URL: {response.get("studio_url")}')
