import aiohttp
import asyncio
import sys


async def sample_request(server_url="http://localhost:3000", token=None):
    url = f"{server_url}/sleep"
    payload = {"seconds": 120}

    async with aiohttp.ClientSession() as session:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with session.post(
            url,
            json=payload,
            headers=headers,
        ) as response:
            result = await response.text()
            assert response.status == 200, (response.status, response.headers, result)
            assert result == "[0]", result


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if len(sys.argv) == 3:
        loop.run_until_complete(sample_request(sys.argv[1], sys.argv[2]))
    elif len(sys.argv) == 2:
        loop.run_until_complete(sample_request(sys.argv[1]))
    else:
        loop.run_until_complete(sample_request())
