import random, string, time
from heapq import nlargest
from loguru import logger

from nltk.data import find as resource_find
from nltk import download as nltk_download

try:
    resource_find('corpora/stopwords')
except LookupError:
    logger.warning("NLTK stopwords not found. Downloading...")
    nltk_download('stopwords')
try:
    resource_find('tokenizers/punkt')
except LookupError:
    logger.warning("NLTK punkt not found. Downloading...")
    nltk_download('punkt')
    
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import tiktoken

import os
import json
from typing import Optional

async def __progressive_summarize_text(text, max_length, initial_reduction_ratio=0.8, step=0.1):
    current_tokens = await __tokenize(text)
    if current_tokens < max_length:
        return text
    
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    sentences = sent_tokenize(text)

    # Remove stopwords and apply stemming
    words = [ps.stem(word) for word in word_tokenize(text.lower()) if word not in stop_words]
    
    # Calculate word frequencies
    word_freqs = FreqDist(words)

    # Score sentences based on word frequency
    sentence_scores = {idx: sum(word_freqs.get(word, 0) for word in word_tokenize(sentence.lower())) for idx, sentence in enumerate(sentences)}
    
    reduction_ratio = initial_reduction_ratio

    # Extract top N sentences based on their scores
    while True:
        num_sentences = max(1, round(len(sentences) * reduction_ratio))
        selected_indexes = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = '\n'.join(sentences[idx] for idx in sorted(selected_indexes))

        if 0 < len(summary.strip()) <= max_length or reduction_ratio - step < 0:
            break
        else:
            reduction_ratio -= step

    return summary

async def __validate_messages_format(messages):
    if not messages:
        return False
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return False
    if messages[0]["role"] == "assistant" or messages[-1]["role"] == "assistant":
        return False
    if any(message["role"] == "system" for message in messages[1:]):
        return False
    return True

async def __split_content(messages):
    text_messages = []
    image_urls = []
    
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    if "text" in item:
                        text_messages.append({"role": message["role"], "content": item["text"]})
                elif item["type"] == "image_url":
                    if "image_url" in item:
                        if isinstance(item["image_url"], str):
                            image_urls.append(item["image_url"])  
                        elif isinstance(item["image_url"], dict) and "url" in item["image_url"]:
                            if item["image_url"]["url"] not in image_urls:
                                image_urls.append(item["image_url"]["url"])
                        else:
                            logger.error(f"Invalid image URL format: {item['image_url']}")
        elif "content" in message and isinstance(message["content"], str):
            text_messages.append({"role": message["role"], "content": message["content"]})  
               
    return text_messages, image_urls

async def __generate_completion_id():
    return "".join(random.choices(string.ascii_letters + string.digits, k=28))

async def __generate_timestamp():
    return int(time.time())

async def __tokenize(text):
    enconder = tiktoken.get_encoding("cl100k_base")
    return len(enconder.encode(text))

async def __stringify_messages(messages):
    return '\n'.join(f"{message['role'].capitalize()}: {message['content']}" for message in messages)

async def __handle_conversation_state(messages: list[dict[str, str]], last_response: str = None, chat_info: dict = None, client_id: str = None) -> tuple[Optional[int], Optional[str], bool]:
    """处理会话状态，返回chatId、chatCode和是否继续会话的标志
    
    当chat_info为None时为查询模式，此时检查会话是否过期并尝试查找匹配的会话
    当chat_info不为None时为保存模式，此时更新或添加会话信息
    """
    conversation_file = "conversation_state.json"
    max_age_minutes = 5
    current_time = time.time()
    
    try:
        # 读取现有会话列表
        conversations = []
        if os.path.exists(conversation_file):
            with open(conversation_file, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        
        # 查询模式：检查是否存在匹配的未过期会话
        if chat_info is None:
            if not messages or not client_id:
                return None, None, False
                
            # 清理过期会话并查找匹配的会话
            updated_conversations = []
            matching_conversation = None
            
            for conv in conversations:
                # 检查会话是否过期
                if (current_time - conv["timestamp"]) / 60 > max_age_minutes:
                    continue
                    
                # 检查是否匹配当前会话
                if (messages[:-1] == conv["context"] and 
                    client_id == conv.get("client_id")):
                    matching_conversation = conv
                
                updated_conversations.append(conv)
            
            # 保存清理后的会话列表
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(updated_conversations, f, ensure_ascii=False, indent=2)
                
            if matching_conversation:
                return (
                    matching_conversation.get("chatId"),
                    matching_conversation.get("chatCode"),
                    True
                )
            return None, None, False
            
        # 保存模式：更新现有会话或添加新会话
        else:
            if not client_id:
                return None, None, False
                
            # 构建新的上下文
            full_context = messages.copy()
            if last_response:
                full_context.append({
                    "role": "assistant",
                    "content": last_response
                })
            
            # 查找是否有匹配的会话需要更新
            found = False
            for conv in conversations:
                if (conv.get("chatId") == chat_info.get("chatId") and 
                    conv.get("chatCode") == chat_info.get("chatCode")):
                    conv["context"] = full_context
                    conv["timestamp"] = current_time
                    found = True
                    break
            
            # 如果没找到匹配的会话，添加新会话
            if not found:
                new_conversation = {
                    "context": full_context,
                    "chatId": chat_info.get("chatId"),
                    "chatCode": chat_info.get("chatCode"),
                    "timestamp": current_time,
                    "client_id": client_id
                }
                conversations.append(new_conversation)
            
            # 保存更新后的会话列表
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
                
            return None, None, False
            
    except Exception as e:
        print(f"处理会话状态时出错: {e}")
        return None, None, False
