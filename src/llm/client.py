#!/usr/bin/env python3
"""LLM client for interacting with Llama endpoint."""

import json
import logging
import time
from typing import Dict, Optional
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with Llama endpoint."""
    
    def __init__(self, endpoint: str = "http://rat.local:8080/v1", max_retries: int = 3):
        """Initialize LLM client.
        
        Args:
            endpoint: Base URL for Llama endpoint
            max_retries: Maximum number of retries for failed requests
        """
        self.endpoint = endpoint.rstrip("/")
        self.max_retries = max_retries
        
    def _call_with_retry(self, messages: list, temperature: float = 0.7) -> Optional[str]:
        """Call LLM endpoint with retry logic.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text if successful, None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json={
                        "model": "llama2",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                    
                logger.warning(f"Unexpected response format: {result}")
                
            except RequestException as e:
                backoff = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                continue
                
        return None
        
    def generate_summary(self, text: str) -> Optional[Dict]:
        """Generate structured summary of conversation text.
        
        Args:
            text: Conversation text to summarize
            
        Returns:
            Dictionary with title, summary and topics if successful,
            None if generation failed
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates structured summaries of conversations."},
            {"role": "user", "content": f"""Please analyze this conversation and provide a JSON response with:
- title: A brief, descriptive title (max 100 chars)
- summary: 2-3 sentence summary of key points (max 500 chars)
- topics: List of 1-5 main topics discussed

Conversation:
{text}

Respond with valid JSON only:
{{"title": "string", "summary": "string", "topics": ["string"]}}"""}
        ]

        response = self._call_with_retry(messages, temperature=0.7)
        if not response:
            return None
            
        try:
            # Parse JSON response
            result = json.loads(response)
            
            # Validate required fields
            required = ["title", "summary", "topics"]
            if not all(k in result for k in required):
                logger.warning(f"Missing required fields in response: {result}")
                return None
                
            # Validate field constraints
            if len(result["title"]) > 100:
                result["title"] = result["title"][:97] + "..."
                
            if len(result["summary"]) > 500:
                result["summary"] = result["summary"][:497] + "..."
                
            if not isinstance(result["topics"], list):
                logger.warning("Topics field is not a list")
                return None
                
            result["topics"] = result["topics"][:5]  # Limit to 5 topics
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response: {e}")
            return None
