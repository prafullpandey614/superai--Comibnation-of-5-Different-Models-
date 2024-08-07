import requests
import replicate,anthropic
import os
from django.conf import settings

class ReplicateClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.replicate.com/v1/predictions"

    def generate_image(self, prompt):
        os.environ['REPLICATE_API_TOKEN'] = settings.REPLICATE_API_KEY
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90
            }
        )
        return output[0]
    
class StableDiffusionClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

    def generate_image(self, prompt, output_format="jpeg"):
        response = requests.post(
            self.base_url,
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "output_format": output_format,
            },
        )
        
        if response.status_code == 200:
            return response.content
        else:
            return None

class HuggingFaceClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct'

    # def generate_text(self, prompt):
    def generate_text(self,prompt, max_length=100):
        data = {
            'inputs': prompt,
            'parameters': {
                'max_length': max_length
            }
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.post(self.base_url , headers=headers, json=data)
        
        if response.status_code == 200:
            generated_text = response.json()
            full_text = generated_text[0]['generated_text']
            if full_text.startswith(prompt):
                full_text = full_text[len(prompt):].strip()
            return full_text
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)
            return None

class ClaudeClient:
    def __init__(self, api_key):
        os.environ['ANTHROPIC_API_KEY'] = settings.CLAUDE_API_KEY
        self.api_key = api_key
        self.base_url = "https://api.claude.ai/generate"

    def generate_response(self, prompt):
        client = anthropic.Anthropic()

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.5,
            system="You are Tutor, Guide in simple words.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        print(message)
        return message.content[0].text


class ZeroGPTClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.zerogpt.com/api/detect/detectText"

    def detect_ai_content(self, text, original_paragraph="string", text_words=0, ai_words=0, fake_percentage=0, sentences=None, h=None, collection_id=0, file_name="string", feedback="string"):
        if sentences is None:
            sentences = []
        if h is None:
            h = []
        
        response = requests.post(
            self.base_url,
            headers={
                "accept": "application/json",
                "ApiKey": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "input_text": text,
                "originalParagraph": original_paragraph,
                "textWords": text_words,
                "aiWords": ai_words,
                "fakePercentage": fake_percentage,
                "sentences": sentences,
                "h": h,
                "collection_id": collection_id,
                "fileName": file_name,
                "feedback": feedback
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
