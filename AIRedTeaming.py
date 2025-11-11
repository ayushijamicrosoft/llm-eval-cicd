from azure.ai.evaluation import evaluate
from azure.ai.evaluation import GroundednessEvaluator
from azure.ai.evaluation.simulator import AdversarialSimulator, AdversarialScenario, DirectAttackSimulator, IndirectAttackSimulator
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
from dataclasses import dataclass
import pandas as pd
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import pprint
from azure.ai.evaluation import ToolCallAccuracyEvaluator , AzureOpenAIModelConfiguration, IntentResolutionEvaluator, TaskAdherenceEvaluator,RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, ViolenceEvaluator, SelfHarmEvaluator, SexualEvaluator, HateUnfairnessEvaluator, CodeVulnerabilityEvaluator, IndirectAttackEvaluator, ProtectedMaterialEvaluator
from pprint import pprint
from azure.storage.blob import BlobServiceClient
import uuid, json

import argparse
import json
import yaml

from azure.keyvault.secrets import SecretClient

file_suffix = uuid.uuid4().hex
print("Run GUID:", file_suffix)

@dataclass
class PromptRecord:
    prompt: str
    simulator: str
    scenario: Optional[str] = None

list_of_prompts = []

# e.g. "https://my-keyvault.vault.azure.net/"
VAULT_URL = "https://eval-agent-kv.vault.azure.net/"

_credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
_secret_client = SecretClient(vault_url=VAULT_URL, credential=_credential)

def get_secret(name: str, *, version: Optional[str] = None) -> str:
    """Return a secret value from Azure Key Vault."""
    return _secret_client.get_secret(name, version=version).value

def load_env_from_keyvault(mapping: dict[str, str]) -> None:
    """
    Given a mapping of ENV_VAR -> KEYVAULT_SECRET_NAME,
    fetch each secret and set os.environ[ENV_VAR].
    """
    for env_var, secret_name in mapping.items():
        value = get_secret(secret_name)
        os.environ[env_var] = value
load_env_from_keyvault({
    "AZURE_OPENAI_ENDPOINT":        "az-openai-endpoint",
    "AZURE_OPENAI_API_KEY":         "az-openai-api-key",
    "AZURE_OPENAI_API_VERSION":     "az-openai-api-version",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "az-openai-deploy",
})

def load_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"⚠️ Config file '{path}' not found. Using defaults.")
        return {}
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f) or {}

def merge_config(defaults: dict, overrides: dict) -> dict:
    """Recursively merge user config over defaults."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in defaults:
            defaults[k] = merge_config(defaults[k], v)
        else:
            defaults[k] = v
    return defaults
def upload_to_blob(container_name: str, file_paths: List[str]) -> None:
    STORAGE_ACCOUNT_NAME = "evalsofagents"
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        pass

    for file_path in file_paths:
        blob_name = os.path.basename(file_path)
        with open(file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"Uploaded: {blob_name}")
# CLI argument for config
parser = argparse.ArgumentParser(description="Run AI Red Teaming with optional config")
parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
args = parser.parse_args()

openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

def create_openai_client() -> AzureOpenAI:
    if not all([OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# Application logic for answering user queries
# ---------------------------------------------------------------------------

def example_application_response(query: str, context: str) -> str:
    client = create_openai_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a user assistant who helps answer questions based on some context.\n\n"
                f"Context: '{context}'"
            ),
        },
        {"role": "user", "content": query},
    ]
    completion = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=10000,
    )
    message = completion.to_dict()["choices"][0]["message"]
    if isinstance(message, dict):
        message = message.content
    return message


async def custom_simulator_callback(
    messages: Dict[str, Any],
    stream: bool = False,
    session_state: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print("Messages", messages)
    
    latest_message = messages[-1]

    application_input = latest_message.content
    context_from_message = latest_message.context
    response = example_application_response(query=application_input, context=context_from_message)

    message = {
        "content": response,
        "role": "assistant",
        "context": context_from_message,
    }
    messages["messages"].append(message)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
        "context": context_from_message,
    }
    
def build_evaluators(
    model_config: AzureOpenAIModelConfiguration,
    credential: DefaultAzureCredential,
) -> Dict[str, Any]:
    if AZURE_AI_PROJECT is None:
        raise RuntimeError("AZURE_AI_PROJECT is not initialized")

    content_safety_kwargs = {"credential": credential, "azure_ai_project": AZURE_AI_PROJECT}

    tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
    intent_resolution = IntentResolutionEvaluator(model_config=model_config)
    task_adherence = TaskAdherenceEvaluator(model_config=model_config)
    relevance = RelevanceEvaluator(model_config=model_config)
    coherence = CoherenceEvaluator(model_config=model_config)
    fluency = FluencyEvaluator(model_config=model_config)

    violence = ViolenceEvaluator(**content_safety_kwargs)
    self_harm = SelfHarmEvaluator(**content_safety_kwargs)
    sexual = SexualEvaluator(**content_safety_kwargs)
    hate_unfairness = HateUnfairnessEvaluator(**content_safety_kwargs)
    code_vulnerability = CodeVulnerabilityEvaluator(**content_safety_kwargs)
    indirect_attack = IndirectAttackEvaluator(**content_safety_kwargs)
    protected_material = ProtectedMaterialEvaluator(**content_safety_kwargs)
    ungrounded_attributes = UngroundedAttributesEvaluator(**content_safety_kwargs)

    evaluator_map: Dict[str, Any] = {
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
        "protected_material": protected_material,
        "ungrounded_attributes": ungrounded_attributes,
    }
    return evaluator_map

print(openai_endpoint)
print(model_name)
print(deployment)
print(api_version)
print(openai_key)

# --- Load defaults from environment / hardcoded ---
defaults = {
    "project": {
    "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
    "project_name": "shayakproject",
    "resource_group_name": "shayak-test-rg",
},
    "simulators": ["adversarial", "direct", "indirect"],
    "evals": [
        "tool_call_accuracy", "intent_resolution", "task_adherence",
        "violence", "relevance", "coherence", "fluency", "self_harm",
        "sexual", "hate_unfairness", "code_vulnerability", "indirect_attack", "protected_material"
    ],
    "key_vault_uri": "",
    "custom_prompts": []
}

# --- Merge user config ---
user_config = load_config(args.config)
config = merge_config(defaults, user_config)

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

subscription_id = config["project"]["subscription_id"]
project_name = config["project"]["project_name"]
resource_group_name = config["project"]["resource_group_name"]

from typing import Optional, Dict, Any
import os

# Azure imports
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy

# OpenAI imports
from openai import AzureOpenAI

# Azure Credential imports
from azure.identity import AzureCliCredential, get_bearer_token_provider

# Initialize Azure credentials
credential = AzureCliCredential()

azure_ai_project = "https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject"

# Azure OpenAI deployment information
azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")  # e.g., "gpt-4"
azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")  # e.g., "your-api-key"
azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")  # Use the latest API version

# Create the `RedTeam` instance with minimal configurations
red_team = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=credential,
    risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness],
    num_objectives=1,
)

import asyncio
import inspect
from pathlib import Path
# ... keep all your other imports and helpers above ...

# Define a callback that uses Azure OpenAI API to generate responses
async def azure_openai_callback(
    messages: list,
    stream: Optional[bool] = False,  # noqa: ARG001
    session_state: Optional[str] = None,  # noqa: ARG001
    context: Optional[Dict[str, Any]] = None,  # noqa: ARG001
) -> dict[str, list[dict[str, str]]]:
    # Get token provider for Azure AD authentication
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_version=azure_openai_api_version,
        azure_ad_token_provider=token_provider,
    )

    ## Extract the latest message from the conversation history
    messages_list = [{"role": message.role, "content": message.content} for message in messages]
    latest_message = messages_list[-1]["content"]

    try:
        # Call the model
        response = client.chat.completions.create(
            model=azure_openai_deployment,
            messages=[
                {"role": "user", "content": latest_message},
            ],
            # max_tokens=500, # If using an o1 base model, comment this line out
            max_completion_tokens=500,  # If using an o1 base model, uncomment this line
            # temperature=0.7, # If using an o1 base model, comment this line out (temperature param not supported for o1 base models)
        )

        # Format the response to follow the expected chat protocol format
        formatted_response = {"content": response.choices[0].message.content, "role": "assistant"}
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e!s}")
        formatted_response = "I encountered an error and couldn't process your request."
    return {"messages": [formatted_response]}

async def main():
    # initialize credential, clients, callbacks etc. here (same as your top-level code)
    credential = AzureCliCredential()
    azure_ai_project = "https://padmajat-agenticai-hack-resource.services.ai.azure.com/api/projects/padmajat-agenticai-hackathon25"
    red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm],
        num_objectives=1,
    )

    # Example target (callback) run
    result = await red_team.scan(
        target=custom_simulator_callback,
        scan_name="Basic-Callback-Scan",
        attack_strategies=[AttackStrategy.Flip],
        output_path="red_team_output.json",
    )
    print("BASIC RESULT -------------------------------------------------------------------------------")
    print(result.attack_details)
    attack_details_list = result.attack_details;
    for attack_detail in attack_details_list:
        list_of_prompts.append(PromptRecord(prompt=attack_detail["conversation"][0]["content"], scenario=attack_detail["attack_technique"], simulator= attack_detail["risk_category"]))
        
    print("Basic scan done:", result)
    print(list_of_prompts)

    # If you want to call a scan that might be synchronous on some versions of the SDK,
    # you can defensively handle both coroutine and normal function:
    # (optional helper)
    async def maybe_await(fn, *args, **kwargs):
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    # Example: run another scan using the helper (handles sync or async scan implementations)
    model_red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm],
        num_objectives=5,
    )
    advanced_result = await maybe_await(
        model_red_team.scan,
        target=azure_openai_callback,
        scan_name="Advanced-Callback-Scan",
        attack_strategies=[AttackStrategy.EASY],  # keep minimal for demo
        output_path=f"Advanced-Callback-Scan_{file_suffix}.json",
    )

    upload_to_blob("advanced-red-teaming-results", f"Advanced-Callback-Scan_{file_suffix}.json")
    print("ADVANCED RESULT -------------------------------------------------------------------------------")
    print(advanced_result.attack_details)
    print("Advanced scan done:", advanced_result)


asyncio.run(main())

'''
async def main2():
    # Define a model configuration to test
    azure_oai_model_config = {
        "azure_endpoint": azure_openai_endpoint,
        "azure_deployment": azure_openai_deployment,
        "api_key": azure_openai_api_key,
    }
    
    # Run the red team scan called "Intermediary-Model-Target-Scan"
    result = await red_team.scan(
        target=azure_oai_model_config,
        scan_name="Intermediary-Model-Target-Scan",
        attack_strategies=[AttackStrategy.Flip],
    )
    risk_category = [RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm]
    for r_cat in risk_category:
        # Create the RedTeam instance with all of the risk categories with 5 attack objectives generated for each category
        model_red_team = RedTeam(
            azure_ai_project=azure_ai_project,
            credential=credential,
            risk_categories=[r_cat],
            num_objectives=20,
        )
        
        # Run the red team scan with multiple attack strategies
        advanced_result = await model_red_team.scan(
            target=azure_openai_callback,
            scan_name="Advanced-Callback-Scan",
            attack_strategies=[
                AttackStrategy.EASY,  # Group of easy complexity attacks
                AttackStrategy.MODERATE,  # Group of moderate complexity attacks
                AttackStrategy.CharacterSpace,  # Add character spaces
                AttackStrategy.ROT13,  # Use ROT13 encoding
                AttackStrategy.UnicodeConfusable,  # Use confusable Unicode characters
                AttackStrategy.CharSwap,  # Swap characters in prompts
                AttackStrategy.Morse,  # Encode prompts in Morse code
                AttackStrategy.Leetspeak,  # Use Leetspeak
                AttackStrategy.Url,  # Use URLs in prompts
                AttackStrategy.Binary,  # Encode prompts in binary
                AttackStrategy.Compose([AttackStrategy.Base64, AttackStrategy.ROT13]),  # Use two strategies in one attack
            ],
            output_path=f"Advanced-Callback-Scan_{r_cat}.json",
        )
        
        path_to_prompts = ".\data\prompts.json"
        
        # Create the RedTeam specifying the custom attack seed prompts to use as objectives
        custom_red_team = RedTeam(
            azure_ai_project=azure_ai_project,
            credential=credential,
            custom_attack_seed_prompts=path_to_prompts,  # Path to a file containing custom attack seed prompts
        )
        
        custom_red_team_result = await custom_red_team.scan(
            target=azure_openai_callback,
            scan_name="Custom-Prompt-Scan",
            attack_strategies=[
                AttackStrategy.EASY,  # Group of easy complexity attacks
                AttackStrategy.MODERATE,  # Group of moderate complexity attacks
                AttackStrategy.DIFFICULT,  # Group of difficult complexity attacks
            ],
            output_path=f"Custom-Prompt-Scan_{r_cat}_{file_suffix}.json",
        )
        print(custom_red_team_results)
        upload_to_blob("custom-red-teaming-results", f"Custom_Prompt-Scan_{r_cat}_{file_suffix}.json")

asyncio.run(main2())
'''
