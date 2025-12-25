import requests
import json
import numpy as np
import base64
from typing import List, Union, Dict, Any

class LLMClient:
    """
    Client for Local LLM (OpenAI-compatible) via LM Studio.
    Defaults to http://localhost:1234/v1
    """
    
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        
        # Default models - User can override or we auto-detect
        self.chat_model = "zai-org/glm-4.6v-flash"
        self.embedding_model = "text-embedding-nomic-embed-text-v1.5@q8_0" 
        # self.embedding_model = "llama-embed-nemotron-8b"
        
        print(f"[LLMClient] Initialized for {self.base_url}")
        print(f"  - Chat Model: {self.chat_model}")
        print(f"  - Emb Model: {self.embedding_model}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get vector embedding for a single text string.
        Returns numpy array of shape (D,).
        """
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": text,
            "model": self.embedding_model
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code != 200:
                print(f"[LLMClient] Error {response.status_code}: {response.text}")
                
            response.raise_for_status()
            data = response.json()
            embedding = data['data'][0]['embedding']
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            # print(f"[LLMClient] Embedding error: {e}")
            raise e

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        Returns numpy array of shape (N, D).
        """
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.embedding_model
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            embeddings_list = [item['embedding'] for item in data['data']]
            return np.array(embeddings_list, dtype=np.float32)
            
        except Exception as e:
            print(f"[LLMClient] Batch embedding error (falling back to sequential): {e}")
            result = []
            for t in texts:
                try:
                    result.append(self.get_embedding(t))
                except:
                    pass
            if result:
                 return np.array(result, dtype=np.float32)
            raise e

    def chat_completion(self, messages: List[Dict[str, str]], temperature=0.7) -> str:
        """
        Simple chat completion.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "messages": messages,
            "model": self.chat_model,
            "temperature": temperature,
            "max_tokens": 1500
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"[LLMClient] Chat error: {e}")
            return ""

    def analyze_image(self, image_path: str, prompt: str = "Analyze this image.") -> str:
        """
        Analyze an image using the Local Vision Model (GLM-4v).
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            payload = {
                "model": self.chat_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.1 # Low temp for analytical observation
            }
            
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                print(f"[LLMClient] Vision Error {response.status_code}: {response.text}")
                return ""
                
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"[LLMClient] Vision failed: {e}")
            return ""

    def check_connection(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=2)
            if resp.status_code == 200:
                models = resp.json()['data']
                model_ids = [m['id'] for m in models]
                print(f"[LLMClient] Available models: {model_ids}")
                
                # Auto-detect logic:
                # If explicit self.embedding_model is NOT in list, try to find a fallback.
                # But since user explicitly set it, we trust it or warn.
                if self.embedding_model not in model_ids:
                     print(f"[LLMClient] WARNING: Configured model {self.embedding_model} not found in {model_ids}")
                
                return True
        except Exception as e:
            print(f"[LLMClient] Connection check failed: {e}")
            return False
        return False
