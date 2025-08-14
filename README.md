# Jupyter Notebook Samples with Tanzu Platform for Cloud Foundry
This repository provides JupyterLab notebooks built with Python, leveraging LangChain and LangGraph. They demonstrate patterns for connecting to a variety of services—such as large language models, data services, and vector databases deployed on to Tanzu Platform for Cloud Foundrywhile incorporating service discovery and credential management.

## Why JupyterLab on TPCF?

### Collaborative Research Platform
- Data Scientists can prototype AI models and analyze results interactively
- ML Engineers can test model integrations and performance
- Application Developers can experiment with AI features before building production apps
- Platform Teams maintain governance and cost control across all usage
### Integrated Workflow Benefits
When JupyterLab is bound to multiple services, you get:
- Data Services connectivity (like Postgres, Valkey, Gemfire, RabbitMQ) for data exploration and feature engineering
- Pre-approved LLM access for natural language processing and generation experiments
- Vector database capabilities (Postgres or Gemfire) for embedding storage and similarity search
- Automatic service discovery and credential management through Tanzu Platform for Cloud Foundry
- Built-in observability for tracking resource usage and model performance


## Getting started
- git clone repository https://github.com/yannicklevederpvtl/cf-jupyterlab-uv
- Add Python Packages
Edit pyproject.toml:
<pre>
langgraph-checkpoint-sqlite
tavily-python
langchain-mcp-adapters
matplotlib
unstructured
pydub
modal
ollama
</pre>

or use command in a JupyterLab Terminal

<pre> 
  uv pip add <package></package> 
</pre>

- Deploy JupyterLab for Cloud Foundry into your TPCF environment using the repository https://github.com/yannicklevederpvtl/cf-jupyterlab-uv
<pre>
  cf push -f manifest.yml
</pre>
- git clone the cf-jupyterlab-samples and start coding
  





