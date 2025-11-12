import asyncio
import os
import uuid
from donkit.ragops_api_gateway_client.client import RagopsAPIGatewayClient

API_URL = os.getenv("RAGOPS_API_GATEWAY_URL", "http://localhost:8080")
API_TOKEN = os.getenv("RAGOPS_API_GATEWAY_TOKEN", "t")
PROJECT_ID = os.getenv("RAGOPS_PROJECT_ID", str(uuid.uuid4()))


async def main():
    async with RagopsAPIGatewayClient(base_url=API_URL, api_token=API_TOKEN) as client:
        # 1. generate
        print("Testing generate...")
        gen_resp = await client.generate(
            provider="openai",
            model_name="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, LLM!"}],
            temperature=0.7,
            max_tokens=32,
            user_id="example-user",
            project_id=PROJECT_ID,
        )
        print("generate response:", gen_resp)

        # 2. generate_stream
        print("Testing generate_stream...")
        async for chunk in client.generate_stream(
            provider="openai",
            model_name="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Stream this!"}],
            temperature=0.7,
            max_tokens=32,
            user_id="example-user",
            project_id=PROJECT_ID,
        ):
            print("stream chunk:", chunk)

        # 3. embeddings
        print("Testing embeddings...")
        emb_resp = await client.embeddings(
            provider="openai",
            input_text="Test embedding input",
            model_name="text-embedding-ada-002",
            user_id="example-user",
            project_id=PROJECT_ID,
        )
        print("embeddings response:", emb_resp)


if __name__ == "__main__":
    asyncio.run(main())
