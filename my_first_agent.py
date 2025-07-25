from pydantic_ai import Agent 
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your api key')
model = GoogleModel('gemini-1.5-flash', provider=provider)

agent = Agent(  
    model,
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)
