import os
from azure.ai.evaluation import evaluate, ToolCallAccuracyEvaluator, AzureOpenAIModelConfiguration
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import EvaluatorIds


openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

project_client = AIProjectClient(
    credential=DefaultAzureCredential(), endpoint=openai_endpoint
)






# Create thread and process user message
thread = project_client.agents.create_thread_and_process_run(agent_id = "asst_PXPfWCnRH0IycicUixpClAK8")
print("should work till here")
#-----------------should work till here ----------------#

project_client.agents.messages.create(thread_id=thread.id, role="user", content="Hello, what Contoso products do you know?")
run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

# Handle run status
if run.status == "failed":
    print(f"Run failed: {run.last_error}")

# Print thread messages
for message in project_client.agents.messages.list(thread_id=thread.id).text_messages:
    print(message)

evaluators={
"Relevance": {"Id": EvaluatorIds.Relevance.value},
"Fluency": {"Id": EvaluatorIds.Fluency.value},
"Coherence": {"Id": EvaluatorIds.Coherence.value},
},

                      
project_client.evaluation.create_agent_evaluation(
    AgentEvaluationRequest(  
        thread=thread.id,  
        run=run.id,   
        evaluators=evaluators,
        appInsightsConnectionString = project_client.telemetry.get_application_insights_connection_string(),
    )
)

from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
import pandas as pd


credential = DefaultAzureCredential()
client = LogsQueryClient(credential)

query = r"""
traces
| where message == "gen_ai.evaluation.result"
| where customDimensions["gen_ai.thread.run.id"] == "{run.id}"
"""

try:
    response = client.query_workspace("DefaultWorkspace-49d64d54-e966-4c46-a868-1999802b762c-EUS", query, timespan=timedelta(days=1))
    if response.status == LogsQueryStatus.SUCCESS:
        data = response.tables
    else:
        # LogsQueryPartialResult - handle error here
        error = response.partial_error
        data = response.partial_data
        print(error)

    for table in data:
        df = pd.DataFrame(data=table.rows, columns=table.columns)
        key_value = df.to_dict(orient="records")
        pprint(key_value)
except HttpResponseError as err:
    print("something fatal happened")
    print(err)

from azure.ai.projects.models import AgentEvaluationRedactionConfiguration
              
project_client.evaluation.create_agent_evaluation(
    AgentEvaluationRequest(  
        thread=thread.id,  
        run=run.id,   
        evaluators=evaluators,  
        redaction_configuration=AgentEvaluationRedactionConfiguration(
            redact_score_properties=False,
       ),
        app_insights_connection_string=app_insights_connection_string,
    )
)

