from azure.ai.evaluation import evaluate
from azure.ai.evaluation import GroundednessEvaluator
from azure.ai.evaluation.simulator import AdversarialSimulator, AdversarialScenario
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.evaluation.simulator import Simulator
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import os
from openai import AzureOpenAI
import importlib.resources as pkg_resources
from dotenv import load_dotenv
import os, json
import pandas as pd
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from user_functions import user_functions
from dotenv import load_dotenv

# Initialize the environment variables for Azure OpenAI and the AI project details
openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

azure_ai_project = {
    "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
    "resource_group_name": "rg-ayushija-2422",
    "project_name": "ayushija-dummy-resource"
}

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=openai_endpoint,
    api_key=openai_key
)

model_config = {
    "azure_endpoint": openai_endpoint,
    "azure_deployment": deployment,
}

# Load the grounding data from the JSON file
resource_name = "grounding.json"
package = "azure.ai.evaluation.simulator._data_sources"
conversation_turns = []

with pkg_resources.path(package, resource_name) as grounding_file, Path(grounding_file).open("r") as file:
    data = json.load(file)

# data = json.load(resource_name);
for item in data:
    conversation_turns.append([item])
    if len(conversation_turns) == 2:
        break
print("---------------------------------------------------------------")
print(conversation_turns)

def example_application_response(query: str, context: str) -> str:
    deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

    # Get a client handle for the AOAI model
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.environ.get("AZURE_API_VERSION"),
        api_key=openai_key
    )

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": f"You are a user assistant who helps answer questions based on some context.\n\nContext: '{context}'",
        },
        {"role": "user", "content": query},
    ]
    # Call the model
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=10000
    )
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(completion)

    print("---------------------------------------------------------------")
    message = completion.to_dict()["choices"][0]["message"]
    if isinstance(message, dict):
        message = message["content"]
    return message

async def custom_simulator_callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    application_input = latest_message["content"]
    context = latest_message.get("context", None)
    # call your endpoint or ai application here
    response = example_application_response(query=application_input, context=context)
    # we are formatting the response to follow the openAI chat protocol format
    message = {
        "content": response,
        "role": "assistant",
        "context": context,
    }
    messages["messages"].append(message)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}
  

custom_simulator =  AdversarialSimulator(credential=DefaultAzureCredential(), azure_ai_project=azure_ai_project)

global list_of_prompts;
list_of_prompts = []
import asyncio

async def main():
    outputs = await custom_simulator(
        scenario=AdversarialScenario.ADVERSARIAL_QA, max_simulation_results=10,
        target=custom_simulator_callback
    )
    
    output_file = "ground_sim_output.jsonl"

    print(outputs)
    with Path(output_file).open("w") as file:
        file.write(outputs.to_eval_qr_json_lines())
    import json

    # assuming outputs.to_eval_qr_json_lines() returns a str with multiple JSON objects separated by newlines
    json_lines = outputs.to_eval_qr_json_lines().splitlines()

    for line in json_lines:
        obj = json.loads(line)
        # adjust the key if it's "query" instead of "prompt"
        list_of_prompts.append(obj.get("query"))
    
    print(list_of_prompts)


asyncio.run(main())


load_dotenv()

from azure.ai.projects import __version__ as projects_version
from packaging.version import Version
# some dependencies have been updated with breaking changes -- indicates whether to use updated models and APIs or not
updated_agents = Version(projects_version) > Version("1.0.0b10") or projects_version.startswith("1.0.0a")

credential=DefaultAzureCredential()

from azure.ai.agents.models import FunctionTool, ToolSet
project_client = AIProjectClient(
    endpoint="https://ayushija-2422-resource.services.ai.azure.com/api/projects/ayushija-2422",
    credential=DefaultAzureCredential()
)

agent = project_client.agents.get_agent(
    model=deployment,
    name="my-assistant",
    instructions="You are a helpful assistant",
    toolset=toolset
)

print(f"Fetched agent, ID: {agent.id}")
thread = project_client.agents.threads.create()
print(f"Created thread, ID: {thread.id}")

# Create message to thread

MESSAGE = "Can you send me an email with weather information for Seattle?"

message = project_client.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content=MESSAGE,
)
    
print(f"Created message, ID: {message.id}")

from azure.ai.agents.models import (
    FunctionTool,
    ListSortOrder,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)
run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)

while run.status in ["queued", "in_progress", "requires_action"]:
    time.sleep(1)
    run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)

    if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        if not tool_calls:
            print("No tool calls provided - cancelling run")
            project_client.agents.runs.cancel(thread_id=thread.id, run_id=run.id)
            break

        tool_outputs = []
        for tool_call in tool_calls:
            if isinstance(tool_call, RequiredFunctionToolCall):
                try:
                    print(f"Executing tool call: {tool_call}")
                    output = functions.execute(tool_call)
                    tool_outputs.append(
                        ToolOutput(
                            tool_call_id=tool_call.id,
                            output=output,
                        )
                    )
                except Exception as e:
                    print(f"Error executing tool_call {tool_call.id}: {e}")

        print(f"Tool outputs: {tool_outputs}")
        if tool_outputs:
            project_client.agents.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
print(f"Run status: {run.status}")

print(f"Run finished with status: {run.status}")

if run.status == "failed":
    print(f"Run failed: {run.last_error}")

print(f"Run ID: {run.id}")


if updated_agents:
    for message in project_client.agents.messages.list(thread.id, order="asc"):
        print(f"Role: {message.role}")
        print(f"Content: {message.content[0].text.value}")
        print("-" * 40)
else:
    for message in project_client.agents.list_messages(thread.id, order="asc").data:
        print(f"Role: {message.role}")
        print(f"Content: {message.content[0].text.value}")
        print("-" * 40)

import json
from azure.ai.evaluation import AIAgentConverter

# Initialize the converter that will be backed by the project.
converter = AIAgentConverter(project_client)

thread_id = thread.id
run_id = run.id

converted_data = converter.convert(thread_id=thread_id, run_id=run_id)
print(json.dumps(converted_data, indent=4))
# Save the converted data to a JSONL file

file_name = "evaluation_data.jsonl"
evaluation_data = converter.prepare_evaluation_data(thread_ids=thread.id, filename=file_name)

load_dotenv()
from azure.ai.evaluation import ToolCallAccuracyEvaluator , AzureOpenAIModelConfiguration, IntentResolutionEvaluator, TaskAdherenceEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, ViolenceEvaluator, SelfHarmEvaluator, SexualEvaluator, HateUnfairnessEvaluator, CodeVulnerabilityEvaluator, IndirectAttackEvaluator, ProtectedMaterialEvaluator
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
    "project_name": "ayushija-dummy-resource",
    "resource_group_name": "rg-ayushija-2422",
}

tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
intent_resolution = IntentResolutionEvaluator(model_config=model_config)
task_adherence = TaskAdherenceEvaluator(model_config=model_config)
relevance = RelevanceEvaluator(model_config=model_config)
coherence = CoherenceEvaluator(model_config=model_config)
fluency = FluencyEvaluator(model_config=model_config)
violence = ViolenceEvaluator(credential=credential, azure_ai_project=azure_ai_project)
self_harm = SelfHarmEvaluator(credential=credential, azure_ai_project=azure_ai_project)
sexual = SexualEvaluator(credential=credential, azure_ai_project=azure_ai_project)
hate_unfairness= HateUnfairnessEvaluator(credential=credential, azure_ai_project=azure_ai_project)
code_vulnerability = CodeVulnerabilityEvaluator(credential=credential, azure_ai_project=azure_ai_project)
indirect_attack = IndirectAttackEvaluator(credential=credential, azure_ai_project=azure_ai_project)
protected_material = ProtectedMaterialEvaluator(credential=credential, azure_ai_project=azure_ai_project)


tool_call_accuracy(query=converted_data['query'], response=converted_data['response'], tool_definitions=converted_data['tool_definitions'])
intent_resolution(query=converted_data['query'], response=converted_data['response'])
task_adherence(query=converted_data['query'], response=converted_data['response'])
violence(query=converted_data['query'], response=converted_data['response'])
relevance(query=converted_data['query'], response=converted_data['response'])
coherence(query=converted_data['query'], response=converted_data['response'])
fluency(response=converted_data['response'])
self_harm(query=converted_data['query'], response=converted_data['response'])
sexual(query=converted_data['query'], response=converted_data['response'])
hate_unfairness(query=converted_data['query'], response=converted_data['response'])
code_vulnerability(query=converted_data['query'], response=converted_data['response'])
indirect_attack(query=converted_data['query'], response=converted_data['response'])
protected_material(query=converted_data['query'], response=converted_data['response'])



from azure.ai.evaluation import evaluate

response = evaluate(
    data=file_name,
    evaluators={
        "tool_call_accuracy": tool_call_accuracy,
        "intent_resolution": intent_resolution,
        "task_adherence": task_adherence,
        "violence": violence,
        "relevance": relevance,
        "coherence": coherence,
        "fluency": fluency,
        "self_harm": self_harm,
        "sexual": sexual,
        "hate_unfairness": hate_unfairness,
        "code_vulnerability": code_vulnerability,
        "indirect_attack": indirect_attack,
        "protected_material": protected_material
    },
    azure_ai_project={
        "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
      "project_name": "ayushija-dummy-resource",
      "resource_group_name": "rg-ayushija-2422",
    }
)
pprint(f'AI Foundary URL: {response.get("studio_url")}')
pprint(response)

# Save evaluation response to JSON file
with open("metrics.json", "w") as f:
    json.dump(response, f, indent=2, default=str)
