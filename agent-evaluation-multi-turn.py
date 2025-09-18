import os, json
import pandas as pd
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from user_functions import user_functions
from dotenv import load_dotenv

openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

load_dotenv()

from azure.ai.projects import __version__ as projects_version
from packaging.version import Version
# some dependencies have been updated with breaking changes -- indicates whether to use updated models and APIs or not
updated_agents = Version(projects_version) > Version("1.0.0b10") or projects_version.startswith("1.0.0a")


project_client = AIProjectClient(
    endpoint="https://padmajat-agenticai-hack-resource.services.ai.azure.com/api/projects/padmajat-agenticai-hackathon25",
    credential=DefaultAzureCredential()
)

'''
Fetch the agent from Azure AI Foundry. 
'''

agent = project_client.agents.get_agent(
    agent_id = "asst_Un6Sw51vsuhlOFjVrW6ksO1D"
)

print(f"Fetched agent, ID: {agent.id}")


'''
Fetch an existing thread Id
'''

thread = project_client.agents.threads.get(thread_id = "thread_Rg1IQ5qBQeF0Z8fvn98oeO0x")
                            
print(f"Fetched thread with thread ID: {thread.id}")

for message in project_client.agents.messages.list(thread.id, order="asc"):
    print(f"Role: {message.role}")
    print(f"Content: {message.content[0].text.value}")
    print("-" * 40)

print("SHOULD WORK TILL HERE AT LEAST-----------------------")

import json
from azure.ai.evaluation import AIAgentConverter

# Initialize the converter that will be backed by the project.
converter = AIAgentConverter(project_client)
thread_id = thread.id

converted_data = converter.convert(thread_id=thread_id, run_id="run_MhHx17PIXfVDvcbil0aM8zrb")
print(json.dumps(converted_data, indent=4))

file_name = "freshEvaluationData.jsonl"
evaluation_data = converter.prepare_evaluation_data(thread_ids=thread.id, filename=file_name)

load_dotenv()
from azure.ai.evaluation import ToolCallAccuracyEvaluator , AzureOpenAIModelConfiguration, IntentResolutionEvaluator, TaskAdherenceEvaluator, ViolenceEvaluator
from pprint import pprint

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=openai_endpoint,
    api_key=openai_key,
    api_version=api_version,
    azure_deployment=deployment,
)
# Needed to use content safety evaluators
azure_ai_project={
    "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
      "project_name": "padmajat-agenticai-hackathon25",
      "resource_group_name": "rg-padmajat-2824",
}

tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
intent_resolution = IntentResolutionEvaluator(model_config=model_config)
task_adherence = TaskAdherenceEvaluator(model_config=model_config)


from azure.ai.evaluation import evaluate

response = evaluate(
    data=file_name,
    evaluators={
        "tool_call_accuracy": tool_call_accuracy,
        "intent_resolution": intent_resolution,
        "task_adherence": task_adherence,
    },
    azure_ai_project={
        "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
      "project_name": "padmajat-agenticai-hackathon25",
      "resource_group_name": "rg-padmajat-2824",
    }
)
pprint(f'AI Foundary URL: {response.get("studio_url")}')
pprint(response)
