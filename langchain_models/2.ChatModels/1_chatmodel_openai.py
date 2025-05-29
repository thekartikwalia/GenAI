from langchain_openai import ChatOpenAI # inherits BaseChatModel class
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o-2024-08-06', temperature=0, max_completion_tokens=10)

result = model.invoke("What is the capital of India?")

print(result) # isn't a plain text
print(result.content) # answer is present in result.content