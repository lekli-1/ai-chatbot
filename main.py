import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv


load_dotenv()

client = AsyncOpenAI()

async def chat(prompt):
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

async def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = await chat(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    asyncio.run(main())
