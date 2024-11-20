import os
import json
import re
from abc import ABC

import openai
from openai.lib.azure import AzureOpenAI
from openai import OpenAI


def is_english(texts):
    eng = 0
    if not texts:
        return False
    for t in texts:
        if re.match(r"[ `a-zA-Z.,':;/\"?<>!\(\)-]", t.strip()):
            eng += 1
    if eng / len(texts) > 0.8:
        return True
    return False


class ModelConfig:
    def __init__(self, model_name, provider, key, base_url):
        self.model_name = model_name
        self.provider = provider
        self.key = key
        self.base_url = base_url

    def create_instance(self):
        if self.provider == "openai":
            return OpenAI_APIChat(self.key, self.model_name, self.base_url)
        elif self.provider == "huggingface":
            return HuggingFaceChat(self.key, self.model_name, self.base_url)
        elif self.provider == "google":
            return GoogleChat(self.key, self.model_name, self.base_url)
        elif self.provider == "azure":
            return AzureChat(self.key, self.model_name, self.base_url)
        elif self.provider == "anthropic":
            return AnthropicChat(self.key, self.model_name, self.base_url)
        else:
            raise ValueError("Provider not supported")


class Base(ABC):
    def __init__(self, key, model_name, base_url):
        timeout = int(os.environ.get("LM_TIMEOUT_SECONDS", 600))
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model_name = model_name

    def chat(self, history, system=None, gen_conf=None):
        if system is not None:
            history.insert(0, {"role": "system", "content": system})
        if gen_conf is None:
            gen_conf = {}
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=history, **gen_conf
            )
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += (
                    "...\nFor the content length reason, it stopped, continue?"
                    if is_english([ans])
                    else "······\n由于长度的原因，回答被截断了，要继续吗？"
                )
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0


class AnthropicChat(Base):
    def __init__(self, key, model_name, base_url=None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=key)
        self.model_name = model_name
        self.system = ""

    def chat(self, system, history, gen_conf):
        if system:
            self.system = system
        if "max_tokens" not in gen_conf:
            gen_conf["max_tokens"] = 4096

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=history,
                system=self.system,
                stream=False,
                **gen_conf,
            ).json()
            ans = response["content"][0]["text"]
            if response["stop_reason"] == "max_tokens":
                ans += (
                    "...\nFor the content length reason, it stopped, continue?"
                    if is_english([ans])
                    else "······\n由于长度的原因，回答被截断了，要继续吗？"
                )
            return (
                ans,
                response["usage"]["input_tokens"] + response["usage"]["output_tokens"],
            )
        except Exception as e:
            return ans + "\n**ERROR**: " + str(e), 0


class AzureChat(Base):
    def __init__(self, key, model_name, **kwargs):
        api_key = json.loads(key).get("api_key", "")
        api_version = json.loads(key).get("api_version", "2024-02-01")
        self.client = AzureOpenAI(
            api_key=api_key, azure_endpoint=kwargs["base_url"], api_version=api_version
        )
        self.model_name = model_name


class GoogleChat(Base):
    def __init__(self, key, model_name, base_url=None):
        from google.oauth2 import service_account
        import base64

        key = json.load(key)
        access_token = json.loads(
            base64.b64decode(key.get("google_service_account_key", ""))
        )
        project_id = key.get("google_project_id", "")
        region = key.get("google_region", "")

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        self.model_name = model_name
        self.system = ""

        if "claude" in self.model_name:
            from anthropic import AnthropicVertex
            from google.auth.transport.requests import Request

            if access_token:
                credits = service_account.Credentials.from_service_account_info(
                    access_token, scopes=scopes
                )
                request = Request()
                credits.refresh(request)
                token = credits.token
                self.client = AnthropicVertex(
                    region=region, project_id=project_id, access_token=token
                )
            else:
                self.client = AnthropicVertex(region=region, project_id=project_id)
        else:
            from google.cloud import aiplatform
            import vertexai.generative_models as glm

            if access_token:
                credits = service_account.Credentials.from_service_account_info(
                    access_token
                )
                aiplatform.init(
                    credentials=credits, project=project_id, location=region
                )
            else:
                aiplatform.init(project=project_id, location=region)
            self.client = glm.GenerativeModel(model_name=self.model_name)

    def chat(self, history, system=None, gen_conf=None):
        if system is not None:
            history.insert(0, {"role": "system", "content": system})
        if gen_conf is None:
            gen_conf = {}
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=history, **gen_conf
            )
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += (
                    "...\nFor the content length reason, it stopped, continue?"
                    if is_english([ans])
                    else "······\n由于长度的原因，回答被截断了，要继续吗？"
                )
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0


class HuggingFaceChat(Base):
    def __init__(self, key=None, model_name="", base_url=""):
        if not base_url:
            raise ValueError("Local llm url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        super().__init__(key, model_name, base_url)


class OpenAI_APIChat(Base):
    def __init__(self, key, model_name, base_url=None):
        if base_url is None:
            base_url = "https://api.openai.com/v1"
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        model_name = model_name.split("___")[0]
        super().__init__(key, model_name, base_url)
