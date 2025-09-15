from azure.ai.evaluation import evaluate
from azure.ai.evaluation import GroundednessEvaluator
from azure.ai.evaluation.simulator import Simulator
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import os
from openai import AzureOpenAI
import importlib.resources as pkg_resources
import os, json
from pathlib import Path
from openai import AzureOpenAI
from azure.ai.evaluation import ContentSafetyEvaluator, evaluate
from azure.ai.evaluation.simulator import AdversarialSimulator, AdversarialScenario
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import os
from openai import AzureOpenAI
import importlib.resources as pkg_resources
import asyncio
import json

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
    "subscription_id": os.environ.get("OIDC_AZURE_SUBSCRIPTION_ID"),
    "resource_group_name": "rg-ayushija-2422",
#    "workspace_name": "ayushija-dummy-resource",
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
# print(data)

# # Option 1: Using open
# with open("grounding.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

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


content_safety_simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)
content_safety_outputs = await content_safety_simulator(
        scenario=content_safety_scenario,
        max_conversation_turns=5,  # define the number of conversation turns
        max_simulation_results=5,  # define the number of simulation results
        target=content_safety_callback,  # define the target model callback
    )

with Path("adv_convo_eval.jsonl").open("w") as f:
    for output in content_safety_outputs:
        f.write(json.dumps({"conversation": output}))
        f.write("\n")

cs_eval = ContentSafetyEvaluator(azure_ai_project=azure_ai_project, credential=credential)

result = evaluate(
    name="content-safety-conversation",
    data="adv_convo_eval.jsonl",
    evaluators={"content_safety": cs_eval},
    # Optionally provide your AI Studio project information to track your evaluation results in your Azure AI Studio project
    azure_ai_project=azure_ai_project,
    # Optionally provide an output path to dump a json of metric summary, row level data and metric and studio URL
    output_path="./content-safety-conversation_results.json",
)
print("Evaluation results: ", result)
