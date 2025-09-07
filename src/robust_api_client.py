#!/usr/bin/env python3
"""
Enhanced API client with robust error handling and retry logic
"""

import time
import logging
import requests
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class APIResponse:
    """Structured API response with metadata"""
    success: bool
    content: str
    response_time_ms: float
    attempt_count: int
    error_message: Optional[str] = None
    rate_limited: bool = False

class RobustAPIClient:
    def call_openai_with_retry(self, prompt: str, config: Dict[str, Any]) -> APIResponse:
        """Make a robust OpenAI API call with retry logic"""
        import json
        url = config.get("api_url", "https://api.openai.com/v1/chat/completions")
        api_key = config.get("api_key")
        model = config.get("model", "gpt-3.5-turbo")
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 512)
        timeout = config.get("timeout", 30)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        attempt = 0
        while attempt <= self.max_retries:
            self._wait_for_rate_limit()
            start_time = time.time()
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
                elapsed = (time.time() - start_time) * 1000
                self.last_request_time = time.time()
                self.requests_made += 1
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    return APIResponse(
                        success=True,
                        content=content,
                        response_time_ms=elapsed,
                        attempt_count=attempt + 1
                    )
                elif response.status_code == 429:
                    # Rate limit
                    wait_time = self._exponential_backoff(attempt)
                    logging.warning(f"OpenAI rate limited, retrying in {wait_time:.2f}s (attempt {attempt+1})")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                else:
                    error_message = f"OpenAI API error {response.status_code}: {response.text}"
                    logging.error(error_message)
                    return APIResponse(
                        success=False,
                        content="",
                        response_time_ms=elapsed,
                        attempt_count=attempt + 1,
                        error_message=error_message,
                        rate_limited=(response.status_code == 429)
                    )
            except requests.RequestException as e:
                elapsed = (time.time() - start_time) * 1000
                logging.error(f"OpenAI API request failed: {e}")
                wait_time = self._exponential_backoff(attempt)
                time.sleep(wait_time)
                attempt += 1
        # If all retries fail
        return APIResponse(
            success=False,
            content="",
            response_time_ms=0.0,
            attempt_count=attempt,
            error_message="OpenAI API call failed after retries",
            rate_limited=False
        )
    """Enhanced API client with retry logic and rate limiting"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", max_retries: int = 3, base_delay: float = 1.0):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.last_request_time = None
        self.min_request_interval = 0.1  # Minimum time between requests (seconds)
        
        # Rate limiting tracking
        self.requests_made = 0
        self.requests_window_start = datetime.now()
        self.max_requests_per_minute = 60
        
    def _wait_for_rate_limit(self):
        """Implement rate limiting"""
        now = datetime.now()
        
        # Reset window if more than a minute has passed
        if now - self.requests_window_start > timedelta(minutes=1):
            self.requests_made = 0
            self.requests_window_start = now
        
        # Check if we've hit the rate limit
        if self.requests_made >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.requests_window_start).total_seconds()
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.requests_made = 0
                self.requests_window_start = datetime.now()
          # Ensure minimum interval between requests
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        return self.base_delay * (2 ** attempt) + (time.time() % 1)  # Add jitter
    
    def call_gemini_with_retry(self, prompt: str, config: Dict[str, Any]) -> APIResponse:
        """Make a robust Gemini API call with retry logic"""
        
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        url = f"{base_url}?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.get("temperature", 0.6),
                "topP": config.get("top_p", 0.95),
                "topK": config.get("top_k", 40),
                "maxOutputTokens": config.get("max_tokens", 1024),
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                self._wait_for_rate_limit()
                
                start_time = time.time()
                
                response = requests.post(
                    url, 
                    json=payload, 
                    headers=headers, 
                    timeout=config.get("timeout", 30)
                )
                
                response_time = (time.time() - start_time) * 1000
                self.last_request_time = time.time()
                self.requests_made += 1
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        
                        return APIResponse(
                            success=True,
                            content=content,
                            response_time_ms=response_time,
                            attempt_count=attempt + 1
                        )
                    except (KeyError, IndexError) as e:
                        logging.error(f"Failed to parse Gemini response: {e}")
                        if attempt == self.max_retries:
                            return APIResponse(
                                success=False,
                                content="",
                                response_time_ms=response_time,
                                attempt_count=attempt + 1,
                                error_message=f"Failed to parse response: {e}"
                            )
                
                elif response.status_code == 429:  # Rate limited
                    rate_limited = True
                    wait_time = self._exponential_backoff(attempt)
                    logging.warning(f"Rate limited, waiting {wait_time:.2f}s before retry {attempt + 1}")
                    
                    if attempt < self.max_retries:
                        time.sleep(wait_time)
                        continue
                    else:
                        return APIResponse(
                            success=False,
                            content="",
                            response_time_ms=response_time,
                            attempt_count=attempt + 1,
                            error_message="Rate limit exceeded",
                            rate_limited=True
                        )
                
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    logging.error(error_msg)
                    
                    if attempt < self.max_retries and response.status_code >= 500:
                        # Retry on server errors
                        wait_time = self._exponential_backoff(attempt)
                        logging.info(f"Server error, retrying in {wait_time:.2f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        return APIResponse(
                            success=False,
                            content="",
                            response_time_ms=response_time,
                            attempt_count=attempt + 1,
                            error_message=error_msg
                        )
                        
            except requests.exceptions.Timeout:
                error_msg = "API request timed out"
                logging.error(error_msg)
                
                if attempt < self.max_retries:
                    wait_time = self._exponential_backoff(attempt)
                    logging.info(f"Timeout, retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)
                    continue
                else:
                    return APIResponse(
                        success=False,
                        content="",
                        response_time_ms=0,
                        attempt_count=attempt + 1,
                        error_message=error_msg
                    )
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logging.error(error_msg)
                
                if attempt < self.max_retries:
                    wait_time = self._exponential_backoff(attempt)
                    logging.info(f"Unexpected error, retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)
                    continue
                else:
                    return APIResponse(
                        success=False,
                        content="",
                        response_time_ms=0,
                        attempt_count=attempt + 1,
                        error_message=error_msg
                    )
        
        # Should never reach here, but just in case
        return APIResponse(
            success=False,
            content="",
            response_time_ms=0,
            attempt_count=self.max_retries + 1,
            error_message="Maximum retries exceeded"
        )

class AdvancedPromptManager:
    """Advanced prompt management with template variations"""
    
    def __init__(self):
        self.generation_templates = {
            "standard": """Based on the following context, please answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:""",
            
            "detailed": """You are a helpful assistant that answers questions based strictly on the provided context. 
Use only the information given in the context to answer the question. If the context doesn't contain 
enough information to answer the question completely, say so explicitly.

Context: {context}

Question: {question}

Please provide a comprehensive answer based solely on the context above:""",
            
            "conservative": """Answer the following question using ONLY the information provided in the context. 
Do not add any information that is not explicitly stated in the context.

Context: {context}

Question: {question}

Answer (based only on the provided context):"""
        }
        
        self.evaluation_templates = {
            "standard": """You are evaluating whether an AI-generated answer contains hallucinations when compared to the provided context.

CONTEXT: {context}

QUESTION: {question}

GENERATED ANSWER: {answer}

EVALUATION CRITERIA:
- A hallucination occurs when the answer contains information NOT supported by the context
- Information that contradicts the context is a hallucination
- Information that goes beyond what's stated in the context is a hallucination
- Reasonable inferences directly supported by the context are NOT hallucinations
- If the answer states "The provided text does not contain information about [topic]" and this is TRUE, then there is NO hallucination
- Answers that honestly state information is missing are FAITHFUL, not hallucinations

SCORING SCALE:
- Faithfulness (1-5): How well does the answer stick to the context?
  1 = Major hallucinations, 2 = Some hallucinations, 3 = Minor issues, 4 = Mostly faithful, 5 = Completely faithful
- Confidence (1-5): How confident are you in this evaluation?
  1 = Very uncertain, 2 = Somewhat uncertain, 3 = Moderately confident, 4 = Confident, 5 = Very confident

IMPORTANT: Answer YES only if there are actual hallucinations (false information). Answer NO if the response is faithful to the context, even if it says information is missing.

Please respond in this exact format:
RESULT|[YES/NO]|[Faithfulness 1-5]|[Confidence 1-5]|[Brief explanation]

Examples: 
- RESULT|NO|5|5|The answer is directly supported by the context.
- RESULT|NO|5|5|The answer correctly states that the context lacks the requested information.""",
            
            "strict": """You are a strict evaluator checking for hallucinations in AI-generated answers.

CONTEXT: {context}
QUESTION: {question}
ANSWER: {answer}

STRICT EVALUATION RULES:
- ANY information not explicitly stated in the context is a hallucination
- ANY inference beyond direct facts is a hallucination
- The answer must be 100% verifiable from the context

Rate faithfulness (1-5) and confidence (1-5), then determine if there are hallucinations (YES/NO).

Format: RESULT|[YES/NO]|[Faithfulness 1-5]|[Confidence 1-5]|[Explanation]""",
              "lenient": """Evaluate this answer for significant hallucinations, allowing reasonable inferences.

CONTEXT: {context}
QUESTION: {question}
ANSWER: {answer}

LENIENT EVALUATION:
- Only mark as hallucination if information clearly contradicts or is unsupported by context
- Reasonable inferences and common knowledge applications are acceptable
- Focus on factual accuracy rather than strict adherence
- Answers stating "information not available" are NOT hallucinations when correct

Rate faithfulness (1-5) and confidence (1-5), then determine if there are significant hallucinations (YES/NO).

Format: RESULT|[YES/NO]|[Faithfulness 1-5]|[Confidence 1-5]|[Explanation]"""
        }
    
    def get_generation_prompt(self, context: str, question: str, template: str = "standard") -> str:
        """Get generation prompt with specified template"""
        template = self.generation_templates.get(template, self.generation_templates["standard"])
        return template.format(context=context, question=question)
    
    def get_evaluation_prompt(self, context: str, question: str, answer: str, template: str = "standard") -> str:
        """Get evaluation prompt with specified template"""
        template = self.evaluation_templates.get(template, self.evaluation_templates["standard"])
        return template.format(context=context, question=question, answer=answer)
