#!/usr/bin/env python3
"""
Simple test script for multimodal (image) requests
"""

import requests
import json


def test_multimodal_requests(base_url="http://localhost:3000"):
    """Test requests with images (base64 and URL format)"""
    print("=== Testing Multimodal Requests (Images) ===")

    # Test 1: Base64 image
    print("Testing base64 image request:")
    payload_base64 = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }
        ],
        "stream": False
    }

    headers = {
        "Content-Type": "application/json",
        "X-OUTPUT-LENGTH": "25"
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload_base64,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Base64 image request successful")
            print(f"  Response: {data['choices'][0]['message']['content'][:60]}...")
            print(f"  Tokens: {data['usage']['total_tokens']}")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 2: URL image
    print("\nTesting URL image request:")
    payload_url = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/cat.png",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload_url,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ URL image request successful")
            print(f"  Response: {data['choices'][0]['message']['content'][:60]}...")
            print(f"  Tokens: {data['usage']['total_tokens']}")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 3: Mixed content (text + multiple images)
    print("\nTesting mixed content (text + multiple images):")
    payload_mixed = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "比较这两张图片"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image1.jpg"}
                    },
                    {"type": "text", "text": "和"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image2.jpg"}
                    }
                ]
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload_mixed,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Mixed content request successful")
            print(f"  Response: {data['choices'][0]['message']['content'][:60]}...")
            print(f"  Tokens: {data['usage']['total_tokens']}")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 4: Streaming with image
    print("\nTesting streaming with image:")
    payload_stream = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "简单描述图片"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }
        ],
        "stream": True
    }

    headers_stream = {
        "Content-Type": "application/json",
        "X-TTFT-MS": "100",
        "X-ITL-MS": "50",
        "X-OUTPUT-LENGTH": "15"
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload_stream,
            headers=headers_stream,
            stream=True
        )

        if response.status_code == 200:
            print(f"  ✓ Streaming with image successful")
            print(f"  Stream response: ", end="")

            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: ') and line_text != 'data: [DONE]':
                        chunk_count += 1
                        try:
                            chunk_data = json.loads(line_text[6:])
                            if chunk_data['choices'][0]['delta'].get('content'):
                                print(chunk_data['choices'][0]['delta']['content'], end='', flush=True)
                        except:
                            pass

            print(f"\n  Chunks received: {chunk_count}")
        else:
            print(f"  ✗ Error: {response.status_code}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print()


def main():
    """Run multimodal tests"""
    print("OpenAI API Emulator - Multimodal Test")
    print("=" * 50)

    base_url = "http://localhost:3000"

    test_multimodal_requests(base_url)

    print("Multimodal test completed!")


if __name__ == "__main__":
    main()