from locust import task, between, HttpUser
import json


class OpenAIEmulatorUser(HttpUser):
    """
    Simulates a user that sends requests to the OpenAI API emulator.
    Tests both streaming and non-streaming chat completions with various timing parameters.
    """

    wait_time = between(1, 3)

    def on_start(self):
        """Initialize test data"""
        self.models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-3.5-turbo-16k"
        ]

        self.sample_messages = [
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "Can you explain quantum computing?"}],
            [{"role": "user", "content": "Write a short story about a robot."}],
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is the meaning of life?"}]
        ]

    @task(3)
    def test_chat_completion_non_stream(self):
        """Test non-streaming chat completion"""
        import random

        payload = {
            "model": random.choice(self.models),
            "messages": random.choice(self.sample_messages),
            "stream": False,
            "max_tokens": random.randint(10, 100)
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": str(random.randint(50, 200)),    # TTFT: 50-200ms
            "X-ITL-MS": str(random.randint(20, 100)),     # ITL: 20-100ms
            "X-OUTPUT-LENGTH": str(random.randint(10, 50)) # Output: 10-50 tokens
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="chat_completion_non_stream",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def test_chat_completion_stream(self):
        """Test streaming chat completion"""
        import random

        payload = {
            "model": random.choice(self.models),
            "messages": random.choice(self.sample_messages),
            "stream": True,
            "max_tokens": random.randint(10, 100)
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": str(random.randint(100, 300)),    # TTFT: 100-300ms
            "X-ITL-MS": str(random.randint(30, 80)),       # ITL: 30-80ms
            "X-OUTPUT-LENGTH": str(random.randint(15, 40)) # Output: 15-40 tokens
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="chat_completion_stream",
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    # Read streaming response
                    chunk_count = 0
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                chunk_count += 1
                                if line_text == 'data: [DONE]':
                                    break

                    if chunk_count > 0:
                        response.success()
                    else:
                        response.failure("No streaming chunks received")
                except Exception as e:
                    response.failure(f"Streaming error: {str(e)}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_models_endpoint(self):
        """Test the models endpoint"""
        with self.client.get(
            "/v1/models",
            name="models_endpoint",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "data" in data and isinstance(data["data"], list):
                        response.success()
                    else:
                        response.failure("Invalid models response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_health_check(self):
        """Test health check endpoint"""
        with self.client.get(
            "/health",
            name="health_check",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and data["status"] == "healthy":
                        response.success()
                    else:
                        response.failure("Unhealthy status")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_timing_parameters(self):
        """Test specific timing parameter scenarios"""
        import random

        # Test high TTFT scenario
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test message"}],
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": "500",  # High TTFT
            "X-ITL-MS": "25",    # Low ITL
            "X-OUTPUT-LENGTH": "15"
        }

        start_time = self.environment.runner.start_time

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="timing_test_high_ttft",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class HighThroughputUser(HttpUser):
    """
    User class for testing high throughput scenarios
    """
    wait_time = between(0.1, 0.5)

    @task
    def rapid_requests(self):
        """Send rapid requests to test concurrency"""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Quick test"}],
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": "50",
            "X-ITL-MS": "10",
            "X-OUTPUT-LENGTH": "5"
        }

        self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="rapid_request"
        )