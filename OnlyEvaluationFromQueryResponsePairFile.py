import os
import sys
import json
import time

from dotenv import load_dotenv
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AIAgentConverter
from azure.storage.blob import BlobClient

from packaging.version import Version
from azure.ai.projects import __version__ as projects_version

from azure.ai.evaluation import (
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
    evaluate,
)

# --------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------


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
    config = load_config()
    credential = DefaultAzureCredential()

    # 1. Get project client
    project_client = get_project_client(
        project_endpoint=config["project_endpoint"],
        credential=credential,
    )

    # 2. Fetch agent
    agent = get_agent(project_client, config["agent_id"])

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
