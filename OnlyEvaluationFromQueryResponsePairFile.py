import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from openai import AzureOpenAI

from azure.ai.evaluation import (
    evaluate,
    ToolCallAccuracyEvaluator,
    AzureOpenAIModelConfiguration,
    IntentResolutionEvaluator,
    TaskAdherenceEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    ViolenceEvaluator,
    SelfHarmEvaluator,
    SexualEvaluator,
    HateUnfairnessEvaluator,
    CodeVulnerabilityEvaluator,
    IndirectAttackEvaluator,
    ProtectedMaterialEvaluator,
    AIAgentConverter,
    UngroundedAttributesEvaluator,
)
from azure.ai.evaluation.simulator import (
    AdversarialSimulator,
    AdversarialScenario,
    DirectAttackSimulator,
    IndirectAttackSimulator,
)
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    FunctionTool,
    ListSortOrder,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)

VAULT_URL = "https://eval-agent-kv.vault.azure.net/"

# --------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------

_credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
_secret_client = SecretClient(vault_url=VAULT_URL, credential=_credential)


def get_secret(name: str, *, version: Optional[str] = None) -> str:
    return _secret_client.get_secret(name, version=version).value


def load_env_from_keyvault(mapping: Dict[str, str]) -> None:
    for env_var, secret_name in mapping.items():
        try:
            value = get_secret(secret_name)
            os.environ[env_var] = value
        except Exception as exc:
            print(f"Warning: could not load secret {secret_name}: {exc}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Config file '{path}' not found. Using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f) or {}


def merge_config(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and k in defaults and isinstance(defaults.get(k), dict):
            defaults[k] = merge_config(defaults[k], v)
        else:
            defaults[k] = v
    return defaults


def default_config() -> Dict[str, Any]:
    return {
        "project": {
            "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
            "project_name": "shayakproject",
            "resource_group_name": "shayak-test-rg",
            "endpoint": "https://shayak-foundry.services.ai.azure.com",
        },
        "agent_id": "asst_OmtWFZGuXJXSfiJ7C41fHDk6",
        "storage_connection_string": "",
        "storage_container": "list-of-thread-ids",
        "storage_blob": "thread_ids_run_ca577c11a2a84c738dfc08d7513f488d.txt",
        "simulators": ["direct", "indirect"],
        "evals": [
            "tool_call_accuracy",
            "intent_resolution",
            "task_adherence",
            "relevance",
            "coherence",
            "fluency",
            "code_vulnerability",
            "indirect_attack",
            #"protected_material",
            "ungrounded_attributes",
        ],
        "key_vault_uri": VAULT_URL,
        "custom_prompts": [
            "Say hello and describe what you are."
        ],
        # quick_mode on by default
        "quick_mode": False,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2E Agent Evaluation with optional config")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Azure clients and helpers
# ---------------------------------------------------------------------------

def init_openai_from_env() -> None:
    global OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_DEPLOYMENT

    load_dotenv()

    load_env_from_keyvault(
        {
            "AZURE_OPENAI_ENDPOINT": "az-openai-endpoint",
            "AZURE_OPENAI_API_KEY": "az-openai-api-key",
            "AZURE_OPENAI_API_VERSION": "az-openai-api-version",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "az-openai-deploy",
        }
    )

    OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
    OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

    print("AOAI endpoint:", OPENAI_ENDPOINT)
    print("AOAI deployment:", OPENAI_DEPLOYMENT)
    print("AOAI api_version:", OPENAI_API_VERSION)


def build_azure_ai_project(config: Dict[str, Any]) -> Dict[str, str]:
    return {
        "subscription_id": config["project"]["subscription_id"],
        "project_name": config["project"]["project_name"],
        "resource_group_name": config["project"]["resource_group_name"],
    }
def build_project_client(config: Dict[str, Any], credential: DefaultAzureCredential) -> AIProjectClient:
    project_name = config["project"]["project_name"]
    endpoint_base = config["project"]["endpoint"].rstrip("/")
    endpoint = f"{endpoint_base}/api/projects/{project_name}"
    return AIProjectClient(endpoint=endpoint, credential=credential)


def build_model_config() -> AzureOpenAIModelConfiguration:
    if not all([OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_DEPLOYMENT]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAIModelConfiguration(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_deployment=OPENAI_DEPLOYMENT,
    )


def create_openai_client() -> AzureOpenAI:
    if not all([OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    )

def load_config():
    """
    Load environment variables and return a config dict.
    Expected env vars:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_CHAT_DEPLOYMENT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION
      - AZURE_AI_PROJECT_ENDPOINT
      - AZURE_AI_PROJECT_NAME
      - AZURE_AI_RESOURCE_GROUP
      - AZURE_AI_SUBSCRIPTION_ID
      - AZURE_AGENT_ID
      - AZURE_STORAGE_CONNECTION_STRING
      - AZURE_STORAGE_CONTAINER_NAME
      - AZURE_STORAGE_BLOB_NAME
    """
    load_dotenv()

    config = {
        "openai_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "deployment": os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "openai_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
        "project_endpoint": os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT",
            # fallback to your hard coded endpoint if not set
            "https://padmajat-agenticai-hack-resource.services.ai.azure.com/api/projects/padmajat-agenticai-hackathon25",
        ),
        "project_name": os.environ.get(
            "AZURE_AI_PROJECT_NAME", "padmajat-agenticai-hackathon25"
        ),
        "resource_group_name": os.environ.get(
            "AZURE_AI_RESOURCE_GROUP", "rg-padmajat-2824"
        ),
        "subscription_id": os.environ.get(
            "AZURE_AI_SUBSCRIPTION_ID", "49d64d54-e966-4c46-a868-1999802b762c"
        ),
        "agent_id": os.environ.get("AZURE_AGENT_ID", "asst_Un6Sw51vsuhlOFjVrW6ksO1D"),
        "storage_connection_string": os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        "storage_container": os.environ.get("AZURE_STORAGE_CONTAINER_NAME"),
        "storage_blob": os.environ.get("AZURE_STORAGE_BLOB_NAME"),
    }

    # propagate to SDK env vars that expect them
    os.environ["AZURE_OPENAI_ENDPOINT"] = config["openai_endpoint"] or ""
    os.environ["AZURE_DEPLOYMENT_NAME"] = config["deployment"] or ""
    os.environ["AZURE_API_VERSION"] = config["api_version"] or ""
    os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

    return config


def is_updated_agents():
    """Flag for whether updated agents API needs to be used."""
    return Version(projects_version) > Version("1.0.0b10") or projects_version.startswith(
        "1.0.0a"
    )


# --------------------------------------------------------------------
# Modular functions for Azure AI Project + Agent
# --------------------------------------------------------------------


def get_project_client(project_endpoint: str, credential=None) -> AIProjectClient:
    """
    Create and return an AIProjectClient.
    """
    if credential is None:
        credential = DefaultAzureCredential()

    client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )
    return client


def get_agent(project_client: AIProjectClient, agent_id: str):
    """
    Fetch and return an agent given its id.
    """
    agent = project_client.agents.get_agent(agent_id=agent_id)
    print(f"Fetched agent, ID: {agent.id}")
    return agent


# --------------------------------------------------------------------
# Fetch thread ids from Azure Blob Storage
# --------------------------------------------------------------------


def get_thread_ids_from_blob(
    connection_string: str, container_name: str, blob_name: str
) -> list[str]:
    """
    Read a txt file from Azure Blob Storage and return a list of thread ids.
    Assumes:
      - One thread id per line in the blob.
    """
    if not all([connection_string, container_name, blob_name]):
        raise ValueError(
            "Storage connection string, container, and blob name must all be set."
        )

    blob_client = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name,
    )

    download_stream = blob_client.download_blob()
    content = download_stream.readall().decode("utf-8")

    thread_ids = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]
    print(f"Fetched {len(thread_ids)} thread ids from blob {container_name}/{blob_name}")
    return thread_ids


# --------------------------------------------------------------------
# Prepare evaluation data
# --------------------------------------------------------------------


def prepare_evaluation_data_file(
    project_client: AIProjectClient,
    thread_ids: list[str],
    output_file: str = "freshEvaluationData.jsonl",
) -> str:
    """
    Use AIAgentConverter to prepare evaluation data jsonl from thread ids.
    """
    converter = AIAgentConverter(project_client)
    evaluation_data = converter.prepare_evaluation_data(
        thread_ids=thread_ids,
        filename=output_file,
    )
    print(f"Wrote evaluation data for {len(thread_ids)} threads to {output_file}")
    return output_file


# --------------------------------------------------------------------
# Set up evaluators and run evaluation
# --------------------------------------------------------------------


def build_evaluators(model_config: AzureOpenAIModelConfiguration, credential, azure_ai_project: dict):
    """
    Create all evaluators in one place.
    """
    tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
    intent_resolution = IntentResolutionEvaluator(model_config=model_config)
    task_adherence = TaskAdherenceEvaluator(model_config=model_config)
    relevance = RelevanceEvaluator(model_config=model_config)
    coherence = CoherenceEvaluator(model_config=model_config)
    fluency = FluencyEvaluator(model_config=model_config)
    violence = ViolenceEvaluator(credential=credential, azure_ai_project=azure_ai_project)
    self_harm = SelfHarmEvaluator(credential=credential, azure_ai_project=azure_ai_project)
    sexual = SexualEvaluator(credential=credential, azure_ai_project=azure_ai_project)
    hate_unfairness = HateUnfairnessEvaluator(
        credential=credential, azure_ai_project=azure_ai_project
    )
    code_vulnerability = CodeVulnerabilityEvaluator(model_config=model_config)
    indirect_attack = IndirectAttackEvaluator(
        credential=credential, azure_ai_project=azure_ai_project
    )
    protected_material = ProtectedMaterialEvaluator(
        credential=credential, azure_ai_project=azure_ai_project
    )

    evaluators = {
        "tool_call_accuracy": tool_call_accuracy,
        "intent_resolution": intent_resolution,
        "task_adherence": task_adherence,
        "relevance": relevance,
        "coherence": coherence,
        "fluency": fluency,
        "violence": violence,
        "self_harm": self_harm,
        "sexual": sexual,
        "hate_unfairness": hate_unfairness,
        "code_vulnerability": code_vulnerability,
        "indirect_attack": indirect_attack,
        "protected_material": protected_material,
    }

    return evaluators


def run_evaluation(data_file: str, config: dict, credential):
    """
    Run azure.ai.evaluation.evaluate on the prepared jsonl data.
    """
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=config["openai_endpoint"],
        api_key=config["openai_key"],
        api_version=config["api_version"],
        azure_deployment=config["deployment"],
    )

    azure_ai_project = {
        "subscription_id": config["subscription_id"],
        "project_name": config["project_name"],
        "resource_group_name": config["resource_group_name"],
    }

    evaluators = build_evaluators(model_config, credential, azure_ai_project)

    response = evaluate(
        data=data_file,
        evaluators=evaluators,
        azure_ai_project=azure_ai_project,
    )

    # studio_url is usually in the top-level response dict
    studio_url = response.get("studio_url")
    if studio_url:
        print(f"AI Foundry URL: {studio_url}")
    else:
        print("Evaluation completed. No studio_url found in response.")

    return response


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main():
    args = parse_args()
    config = merge_config(default_config(), load_config(args.config))
    credential = DefaultAzureCredential()

    init_openai_from_env()

    AZURE_AI_PROJECT = build_azure_ai_project(config)
    print("Azure AI project:", AZURE_AI_PROJECT)

    credential = DefaultAzureCredential()

    project_client = build_project_client(config, credential)
    agent = project_client.agents.get_agent(agent_id=config["agent_id"])
    # 3. Fetch thread ids from blob txt file
    thread_ids = get_thread_ids_from_blob(
        connection_string=config["storage_connection_string"],
        container_name=config["storage_container"],
        blob_name=config["storage_blob"],
    )

    # 4. Convert thread ids to evaluation data jsonl
    data_file = prepare_evaluation_data_file(
        project_client=project_client,
        thread_ids=thread_ids,
        output_file="freshEvaluationData.jsonl",
    )

    # 5. Run evaluation
    try:
        run_evaluation(data_file=data_file, config=config, credential=credential)
    except Exception as exp:
        # fall back to printing the exception
        print(f"Evaluation failed with error: {exp}")


if __name__ == "__main__":
    main()
