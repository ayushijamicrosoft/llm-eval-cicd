import os
from azure.ai.evaluation import evaluate, ToolCallAccuracyEvaluator, AzureOpenAIModelConfiguration
from azure.ai.project import AIProjectClient

openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

# -------------------------------------------------------------------
# 1. Setup Model Config (Azure OpenAI)
# -------------------------------------------------------------------
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_API_VERSION"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
)

tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)

# -------------------------------------------------------------------
# 2. Connect to AI Foundry Project
# -------------------------------------------------------------------
subscription_id = "49d64d54-e966-4c46-a868-1999802b762c"
resource_group = "rg-ayushija-2422"
project_name = "ayushija-dummy-resource"

# Create project client (needs correct auth; DefaultAzureCredential works too)
client = AIProjectClient(
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    project_name=project_name,
)

# -------------------------------------------------------------------
# 3. Create Dataset Automatically from Agent Runs
# -------------------------------------------------------------------
# This grabs the most recent runs for a given agent
dataset = client.datasets.create_from_agent_runs(
    name="latest_agent_dataset",             # dataset name (auto-created if not exists)
    run_filter={"agent_name": "Agent396"},   # replace with your agent's name in Foundry
)

print(f"✅ Dataset created from agent runs: {dataset.name}")

# -------------------------------------------------------------------
# 4. Run Evaluation against the Auto Dataset
# -------------------------------------------------------------------
response = evaluate(
    evaluation_name="Tool Call Accuracy Evaluation",
    evaluators={"tool_call_accuracy": tool_call_accuracy},
    azure_ai_project={
        "subscription_id": subscription_id,
        "project_name": project_name,
        "resource_group_name": resource_group,
    },
    dataset_name=dataset.name,  # 👈 directly use dataset from runs
)

print(f"📊 AI Foundry Results: {response.get('studio_url')}")
