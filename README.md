# Evaluation of Agents


In this project, we utilize **Azure AI Foundry Python SDK** to evaluate AI agents on various parameters like Groundedness, Relevance, Safety, Tool Call Adherance.

There are three main stages: 

1. Generation of synthetic prompts for different scenarios like direct/indirect jail break attacks, advarsarial prompts - using Simulator. User can add their own custom prompts to this list as well. 
2. Execution of multi-turn conversations with prompts to the agent - every conversation is called as "thread"
3. Convertion of every conversation to the correct format (using AI Agent converter function) for evaluation with different evaluators.

## How to run the repository?

### Workflows
1. GitHub Actions - are used to run the python files, everytime a change is pushed to the main branch. Refer to the .workflows/ folder for the yml file.
2. The repository secrets are stored in Github settings > Secrets
3. Output of every workflow is visible in **Actions** tab.
4. You can download the final output file from the workflow run.
5. The final output file is in the form of *.jsonl.
6. Additionally, the logs are visible in Azure ML Studio link in Azure Portal.

### Use the output file
1. Go to Azure ML Studio > Evaluations tab > Logs + Metrics
2. Pick up the **instance_results.jsonl** file and copy paste the contents into a **instance_results.json** file. 
3. Locally, convert this **instance_results.json** file into **converted_data.jsonl** file by running the python file **format_output.py** from this GitHub Repo.
4. Now, upload this **converted_data.jsonl** file to the Azure AI Foundry > Evaluation > Create New Evaluation.
5. In Azur AI Studio Evals - you can choose which evals you want to reflect.Additionally, Azure ML studio has those same details as well.

### List of Files to use. 
1. To run the end to end flow: **EndtoEndAgentEvaluation.py**
2. To convert the logs in Azure ML Studio to Azure AI Evals results: **format_output.py**
3. To run the workflows: **.github/.workflows/**

### PS:

1. The python file runs only for the GitHub Actions, and would not run locally - this is because of a setup on Azure.
2. Azure Foundry Evals is currently in Preview stage - so there could be more changes, but the idea of the project is to run an overall flow. 

