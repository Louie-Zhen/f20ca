"""
LLM Provider utilities for Garage Booking Assistant
Supports multiple LLM providers: OpenRouter, Cohere
"""
import logging
import requests
import json
import cohere
from groq import Groq

logger = logging.getLogger(__name__)

def get_llm_response_groq(user_message, api_key, system_message=None):
    """Get response from Cohere API"""
    try:
        client = Groq(api_key=api_key)

        if system_message is None:
            system_message = "You are a helpful garage booking assistant. Help users book garage appointments, check availability, and answer questions about garage services. Be concise and friendly."

        stream = client.chat.completions.create(
            messages=[
                # Set an optional system message. This sets the behavior of the
                # assistant and can be used to provide specific instructions for
                # how it should behave throughout the conversation.
                {
                    "role": "system",
                    "content": system_message
                },
                # Set a user message for the assistant to respond to.
                {
                    "role": "user",
                    "content": user_message,
                }
            ],

            # The language model which will generate the completion.
            model="llama-3.1-8b-instant",
            max_completion_tokens=200,
            stream=False,
        )
        """
        full_response = ""

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content

        return full_response
        """
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        raise

def get_llm_response_openrouter(user_message, api_key):
    """Get response from OpenRouter API"""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Garage Booking Assistant",
            },
            data=json.dumps({
                "model": "qwen/qwen3-4b:free",
                "messages": [
                    {
                        "role": "user",
                        "content": f"You are a helpful garage booking assistant. Help users book garage appointments, check availability, and answer questions about garage services. Be concise and friendly.\n\nUser: {user_message}\nAssistant:"
                    }
                ],
                "max_tokens": 500
            }),
            timeout=30
        )

        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']

    except requests.exceptions.HTTPError as e:
        logger.error(f"OpenRouter HTTP Error: {e}")
        try:
            error_detail = response.json()
            logger.error(f"API Error details: {error_detail}")
        except:
            logger.error(f"Response text: {response.text}")
        raise
    except Exception as e:
        logger.error(f"OpenRouter Error: {e}")
        raise


def get_llm_response_cohere(user_message, api_key, system_message=None):
    """Get response from Cohere API"""
    try:
        co = cohere.ClientV2(api_key)

        if system_message is None:
            system_message = "You are a helpful garage booking assistant. Help users book garage appointments, check availability, and answer questions about garage services. Be concise and friendly."

        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500
        )

        return response.message.content[0].text

    except Exception as e:
        logger.error(f"Cohere Error: {e}")
        raise


def build_booking_system_prompt(booking_state):
    """
    Build a conversation-aware system prompt
    Uses history instead of tracking individual fields - simpler and faster
    """
    conversation_history = booking_state.get_conversation_history()

    system_prompt = f"""You are a garage booking assistant. Your job is to collect booking information efficiently.Your ONLY job is to collect specific information.

INFORMATION NEEDED (in order):
1. Full name
2. Car registration number
3. Car make and model
4. Current mileage
5. Service contract/warranty? (yes/no)
6. What service or issue brings them in

CONVERSATION SO FAR:
{conversation_history}

STRICT RESPONSE RULES:
1. MAX LENGTH: 20 words. 
2. NO PLEASANTRIES: Do not say "Got it", "Thank you", "I understand", or "You mentioned...".
3. NO SUMMARIES: DO NOT repeat what the user just said.
4. ONE TASK: Just ask the single next missing question directly.

GOOD EXAMPLE: "Is your car currently under any warranty?"
BAD EXAMPLE: "Got it, you need an oil change. Now, do you have a warranty?"
"""

    return system_prompt


def get_llm_response(user_message, provider, openrouter_key=None, cohere_key=None,groq_key=None, system_message=None):
    """Get response from configured LLM provider"""
    try:
        logger.info(f"Sending to {provider.upper()}: {user_message}")

        if provider == 'openrouter':
            llm_response = get_llm_response_openrouter(user_message, openrouter_key)
        elif provider == 'cohere':
            llm_response = get_llm_response_cohere(user_message, cohere_key, system_message=system_message)
        elif provider == 'groq':
            llm_response = get_llm_response_groq(user_message, groq_key, system_message=system_message)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        logger.info(f"LLM response: {llm_response}")
        return llm_response

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again."
