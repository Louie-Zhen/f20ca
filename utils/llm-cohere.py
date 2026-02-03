"""
LLM Provider utilities for Garage Booking Assistant
Supports multiple LLM providers: OpenRouter, Cohere
"""
import logging
import requests
import json
import cohere
from cohere import SystemChatMessageV2, UserChatMessageV2
import os

logger = logging.getLogger(__name__)

# ==============================================================================
#  GLOBAL LANGUAGE CONFIGURATION (全局语言配置区)
#  修改这里的变量即可改变整个系统的语言设定
# ==============================================================================

# 1. 默认系统人设 (用于简单的通用对话)
#DEFAULT_SYSTEM_MESSAGE = "你是一个乐于助人的车库预订助手。帮助用户预订车库预约，检查可用性，并回答有关车库服务的问题。请保持简洁友好，并始终使用中文回复。"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful garage booking assistant. Help users book garage appointments, check availability, and answer questions about garage services. Be concise and friendly."

# 2. 预订流程的详细提示词模板
# 注意：请保留 {conversation_history} 这个占位符，程序会自动填入历史记录

# BOOKING_PROMPT_TEMPLATE = """你是一个车库预订助手。你的工作是高效地收集预订信息。
# 请完全使用中文与用户交流。
#
# 需要收集的信息（必须严格按以下顺序，每次只问一个）：
# 1. 客户全名 (Full name)
# 2. 车牌号 (Car registration number)
# 3. 汽车品牌和型号 (Car make and model)
# 4. 当前里程数 (Current mileage)
# 5. 是否有服务合同/保修？(Service contract/warranty - yes/no)
# 6. 需要什么服务或遇到了什么问题 (What service or issue)
#
# 目前的对话历史：
# {conversation_history}
#
# 规则：
# - 保持简洁——每次回答最多 2 个短句。
# - 每次只询问 **一条** 缺失的信息。
# - 必须严格按照上述顺序询问。
# - 永远不要重复询问已经知道的信息——请先检查上面的对话历史。
# - 在收集完所有 6 条信息之前，不要询问日期/时间。
# - 一旦收集齐所有 6 条信息，告诉用户你将查询可用日期。
# - 不要闲聊——专注于任务。
# """

BOOKING_PROMPT_TEMPLATE = """You are a garage booking assistant. Your job is to collect booking information efficiently.

INFORMATION NEEDED (in order):
1. Full name
2. Car registration number
3. Car make and model
4. Current mileage
5. Service contract/warranty? (yes/no)
6. What service or issue brings them in

CONVERSATION SO FAR:
{conversation_history}

RULES:
- Be concise - max 2 short sentences
- Ask for ONE missing piece of information at a time
- Follow the order above
- Never repeat questions - check the conversation history
- Don't ask for date/time until all 6 pieces above are collected
- Once you have all 6, say you'll check available dates
- Don't be chatty - stay focused on the task"""

# 3. 当系统出错时的兜底回复
#ERROR_MESSAGE_TEXT = "抱歉，我现在处理您的请求时遇到了一些问题，请稍后再试。"
ERROR_MESSAGE_TEXT = "I'm sorry, I'm having trouble processing your request right now. Please try again."

# 4. Cohere 模型选择 (全局设置)
COHERE_MODEL_NAME = "command-r-08-2024"#"command-light"

# ==============================================================================


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
                "model": "qwen/qwen-2.5-72b-instruct", # 建议使用支持中文更好的模型
                "messages": [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_MESSAGE # 使用全局变量
                    },
                    {
                        "role": "user",
                        "content": user_message
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

        # 如果没有传入特定的 system_message，则使用全局默认值
        if system_message is None:
            system_message = DEFAULT_SYSTEM_MESSAGE

        response = co.chat(
            model=COHERE_MODEL_NAME, # 使用全局变量
            messages=[
                SystemChatMessageV2(content=system_message),
                UserChatMessageV2(content=user_message)
            ],
            #限制最大输出长度
            max_tokens=100
        )

        return response.message.content[0].text

    except Exception as e:
        logger.error(f"Cohere Error: {e}")
        raise


def build_booking_system_prompt(booking_state):
    """
    Build a conversation-aware system prompt
    Uses the global template to insert history
    """
    conversation_history = booking_state.get_conversation_history()

    # 使用全局模板并填入历史记录
    system_prompt = BOOKING_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history
    )

    return system_prompt


def get_llm_response(user_message, provider, openrouter_key=None, cohere_key=None, system_message=None):
    """Get response from configured LLM provider"""
    try:
        logger.info(f"Sending to {provider.upper()}: {user_message}")

        if provider == 'openrouter':
            llm_response = get_llm_response_openrouter(user_message, openrouter_key)
        elif provider == 'cohere':
            llm_response = get_llm_response_cohere(user_message, cohere_key, system_message=system_message)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        logger.info(f"LLM response: {llm_response}")
        return llm_response

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        # 使用全局错误提示变量
        return ERROR_MESSAGE_TEXT