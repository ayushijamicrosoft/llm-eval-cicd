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


openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

print(os.environ.get("OIDC_AZURE_SUBSCRIPTION_ID"))
openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

async def content_safety_callback(
    messages: List[Dict], stream: bool = False, session_state: Optional[str] = None, context: Optional[Dict] = None
) -> dict:
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    # Get a client handle for the model
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=openai_endpoint,
        api_key=openai_key
    )
    # Call the model
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "user",
                    "content": messages["messages"][0]["content"],
                }
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        formatted_response = completion.to_dict()["choices"][0]["message"]
    except Exception:
        formatted_response = {
            "content": "I don't know",
            "role": "assistant",
            "context": {"key": {}},
        }
    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
        "context": context,
    }

async def main():
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
    resource_name = "grounding.json"
    package = "azure.ai.evaluation.simulator._data_sources"
    conversation_turns: List[List[Dict[str, Any]]] = []

    with pkg_resources.path(package, resource_name) as grounding_file, Path(grounding_file).open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Use first 2 items for a tiny demo
    for item in data:
        conversation_turns.append([item])
        if len(conversation_turns) == 2:
            break
            
    credential=DefaultAzureCredential()
    content_safety_simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)

    content_safety_scenario = AdversarialScenario.ADVERSARIAL_CONVERSATION
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
    print(credential)
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

if __name__ == "__main__":
    asyncio.run(main())
