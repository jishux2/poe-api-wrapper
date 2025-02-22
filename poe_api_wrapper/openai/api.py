from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from daphne.cli import CommandLineInterface
from typing import Any, Union, AsyncGenerator
from poe_api_wrapper import AsyncPoeApi
from poe_api_wrapper.openai import helpers
from poe_api_wrapper.openai.type import *
import orjson, asyncio, random, os, uuid
from httpx import AsyncClient
import json

# 定义全局变量
remaining_balance = 0
current_conversation_cost = 0

DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Poe API Wrapper", description="OpenAI Proxy Server")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

with open(os.path.join(DIR, "secrets.json"), "rb") as f:
    TOKENS = orjson.loads(f.read())
    if "tokens" not in TOKENS:
        raise Exception("Tokens not found in secrets.json")
    app.state.tokens = TOKENS["tokens"]

with open(os.path.join(DIR, "models.json"), "rb") as f:
    models = orjson.loads(f.read())
    app.state.models = models

@app.get("/", response_model=None)
async def index() -> ORJSONResponse:
    return ORJSONResponse({"message": "Welcome to Poe Api Wrapper reverse proxy!",
                            "docs": "See project docs @ https://github.com/snowby666/poe-api-wrapper"})


@app.api_route("/models/{model}", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
@app.api_route("/models/{model}", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
@app.api_route("/models", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
@app.api_route("/v1/models", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
async def list_models(request: Request, model: str = None) -> ORJSONResponse:
    if model:
        if model not in app.state.models:
            raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
        return ORJSONResponse({"id": model, "object": "model", "created": await helpers.__generate_timestamp(), "owned_by": app.state.models[model]["owned_by"], "tokens": app.state.models[model]["tokens"], "endpoints": app.state.models[model]["endpoints"]})
    modelsData = [{"id": model, "object": "model", "created": await helpers.__generate_timestamp(), "owned_by": values["owned_by"], "tokens": values["tokens"], "endpoints": values["endpoints"]} for model, values in app.state.models.items()]
    return ORJSONResponse({"object": "list", "data": modelsData})


@app.api_route("/chat/completions", methods=["POST", "OPTIONS"], response_model=None)
@app.api_route("/v1/chat/completions", methods=["POST", "OPTIONS"], response_model=None)
async def chat_completions(request: Request, data: ChatData) -> Union[StreamingResponse, ORJSONResponse]:
    messages, model, streaming = data.messages, data.model, data.stream
    max_tokens, stream_options = data.max_tokens, data.stream_options
    enable_tracking = data.enable_conversation_tracking  # 获取控制变量

    # Validate messages format
    if not await helpers.__validate_messages_format(messages):
        raise HTTPException(detail={"error": {"message": "Invalid messages format.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if model not in app.state.models:
        raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    include_usage = stream_options.get("include_usage", False) if stream_options else False
    
    modelData = app.state.models[model]
    baseModel, tokensLimit = modelData["baseModel"], modelData["tokens"]
    endpoints, premiumModel = modelData["endpoints"], modelData["premium_model"]
    
    if "/v1/chat/completions" not in endpoints:
        raise HTTPException(detail={"error": {"message": "This model does not support chat completions.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    client, subscription, client_id = await rotate_token(app.state.tokens)
    
    if premiumModel and not subscription:
        raise HTTPException(detail={"error": {"message": "Premium model requires a subscription.", "type": "error", "param": None, "code": 402}}, status_code=402)
    
    text_messages, image_urls = await helpers.__split_content(messages)
    
    response = await message_handler(baseModel, text_messages, tokensLimit, enable_tracking, client_id)
    prompt_tokens = await helpers.__tokenize(''.join([str(message) for message in response["message"]]))
    if max_tokens and sum((max_tokens, prompt_tokens)) > app.state.models[model]["tokens"]:
        raise HTTPException(detail={"error": {
                                        "message": f"This model's maximum context length is {app.state.models[model]['tokens']} tokens. However your request exceeds this limit ({max_tokens} in max_tokens, {prompt_tokens} in messages).", 
                                        "type": "error", 
                                        "param": None, 
                                        "code": 400}
                                    }, status_code=400)
            
    completion_id = await helpers.__generate_completion_id()
    
    return await streaming_response(client, response, model, completion_id, prompt_tokens, image_urls, max_tokens, include_usage, text_messages, enable_tracking, client_id) \
        if streaming else await non_streaming_response(client, response, model, completion_id, prompt_tokens, image_urls, max_tokens)


@app.api_route("/images/generations", methods=["POST", "OPTIONS"], response_model=None)
@app.api_route("/v1/images/generations", methods=["POST", "OPTIONS"], response_model=None)
async def create_images(request: Request, data: ImagesGenData) -> ORJSONResponse:
    prompt, model, n, size = data.prompt, data.model, data.n, data.size
    
    if not isinstance(prompt, str):
        raise HTTPException(detail={"error": {"message": "Invalid prompt.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if model not in app.state.models:
        raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if not isinstance(n, int) or n < 1:
        raise HTTPException(detail={"error": {"message": "Invalid n value.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if size == "1024x1024":
        aspect_ratio = ""
    elif "sizes" in app.state.models[model] and size in app.state.models[model]["sizes"]:
        aspect_ratio = app.state.models[model]["sizes"][size]
    else:
        raise HTTPException(detail={"error": {"message": f"Invalid size for {model}. Available sizes: {', '.join(app.state.models[model]['sizes']) if 'sizes' in app.state.models[model] else '1024x1024'}", "type": "error", "param": None, "code": 400}}, status_code=400)

    modelData = app.state.models[model]
    baseModel, tokensLimit, endpoints, premiumModel = modelData["baseModel"], modelData["tokens"], modelData["endpoints"], modelData["premium_model"]
    
    if "/v1/images/generations" not in endpoints:
        raise HTTPException(detail={"error": {"message": "This model does not support image generation.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    client, subscription = await rotate_token(app.state.tokens)
    
    if premiumModel and not subscription:
        raise HTTPException(detail={"error": {"message": "Premium model requires a subscription.", "type": "error", "param": None, "code": 402}}, status_code=402)
    
    response = await image_handler(baseModel, prompt, tokensLimit)
    
    urls = []
    for _ in range(n):
        image_generation = await generate_image(client, response, aspect_ratio)
        urls.extend([url for url in image_generation.split() if url.startswith("https://")])
        if len(urls) >= n:
            break
    urls = urls[-n:]
    
    if len(urls) == 0:
        raise HTTPException(detail={"error": {"message": f"The provider for {model} sent an invalid response.", "type": "error", "param": None, "code": 500}}, status_code=500)
        
    async with AsyncClient(http2=True) as fetcher:
        for url in urls:
            r = await fetcher.get(url)
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(detail={"error": {"message": "The content returned was not an image.", "type": "error", "param": None, "code": 500}}, status_code=500)

    return ORJSONResponse({"created": await helpers.__generate_timestamp(), "data": [{"url": url} for url in urls]})


@app.api_route("/images/edits", methods=["POST", "OPTIONS"], response_model=None)
@app.api_route("/v1/images/edits", methods=["POST", "OPTIONS"], response_model=None)
async def edit_images(request: Request, data: ImagesEditData) -> ORJSONResponse:
    image, prompt, model, n, size = data.image, data.prompt, data.model, data.n, data.size
    
    if not (isinstance(image, str) and (os.path.exists(image) or image.startswith("http"))):
        raise HTTPException(detail={"error": {"message": "Invalid image.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if not isinstance(prompt, str):
        raise HTTPException(detail={"error": {"message": "Invalid prompt.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if model not in app.state.models:
        raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if not isinstance(n, int) or n < 1:
        raise HTTPException(detail={"error": {"message": "Invalid n value.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if size == "1024x1024":
        aspect_ratio = ""
    elif "sizes" in app.state.models[model] and size in app.state.models[model]["sizes"]:
        aspect_ratio = app.state.models[model]["sizes"][size]
    else:
        raise HTTPException(detail={"error": {"message": f"Invalid size for {model}. Available sizes: {', '.join(app.state.models[model]['sizes']) if 'sizes' in app.state.models[model] else '1024x1024'}", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    modelData = app.state.models[model]
    baseModel, tokensLimit, endpoints, premiumModel = modelData["baseModel"], modelData["tokens"], modelData["endpoints"], modelData["premium_model"]
    
    if "/v1/images/edits" not in endpoints:
        raise HTTPException(detail={"error": {"message": "This model does not support image editing.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    client, subscription = await rotate_token(app.state.tokens)
    
    if premiumModel and not subscription:
        raise HTTPException(detail={"error": {"message": "Premium model requires a subscription.", "type": "error", "param": None, "code": 402}}, status_code=402)
    
    response = await image_handler(baseModel, prompt, tokensLimit)
    
    urls = []
    for _ in range(n):
        image_generation = await generate_image(client, response, aspect_ratio, [image])
        urls.extend([url for url in image_generation.split() if url.startswith("https://")])
        if len(urls) >= n:
            break
    urls = urls[-n:]
        
    if len(urls) == 0:
        raise HTTPException(detail={"error": {"message": f"The provider for {model} sent an invalid response.", "type": "error", "param": None, "code": 500}}, status_code=500)
    
    async with AsyncClient(http2=True) as fetcher:
        for url in urls:
            r = await fetcher.get(url)
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(detail={"error": {"message": "The content returned was not an image.", "type": "error", "param": None, "code": 500}}, status_code=500)
            
    return ORJSONResponse({"created": await helpers.__generate_timestamp(), "data": [{"url": url} for url in urls]})
   

async def image_handler(baseModel: str, prompt: str, tokensLimit: int) -> dict:
    try:
        message = await helpers.__progressive_summarize_text(prompt, min(len(prompt), tokensLimit))
        return {"bot": baseModel, "message": message}
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to truncate prompt. Error: {e}", "type": "error", "param": None, "code": 400}}, status_code=400) from e
   
   
async def message_handler(
    baseModel: str, messages: list[dict[str, str]], tokensLimit: int, 
    enable_tracking: bool = False, client_id: str = None
) -> dict:
    try:
        # 如果启用了会话追踪，先检查是否需要继续之前的会话
        chat_id = None
        chat_code = None
        continue_conversation = False
        
        if enable_tracking:
            chat_id, chat_code, continue_conversation = await helpers.__handle_conversation_state(messages=messages, client_id=client_id)
        
        if continue_conversation:
            # 直接使用最新的消息
            main_request = messages[-1]['content']
            return {
                "bot": baseModel,
                "message": main_request,
                "chatId": chat_id,
                "chatCode": chat_code
            }
        
        # 原有的处理逻辑
        system_message = ""
        if messages[0]['role'] == 'system':
            system_message = f"System: {messages[0]['content']}\n\n"
            system_tokens = await helpers.__tokenize(system_message)
            messages = messages[1:]
        else:
            system_tokens = 0

        reversed_messages = list(reversed(messages))
        included_messages = []
        total_tokens = system_tokens
        
        for idx, msg in enumerate(reversed_messages):
            msg_string = f"{msg['role'].capitalize()}: {msg['content']}\n\n"
            msg_tokens = await helpers.__tokenize(msg_string)
            
            if total_tokens + msg_tokens > tokensLimit:
                break
            
            included_messages.append(msg_string)
            total_tokens += msg_tokens
        
        included_messages.reverse()
        main_request = included_messages.pop()
        history_string = system_message + ''.join(included_messages)
        message = f"Your current message context: \n{history_string}The most recent message: {main_request}\n"
        
        history_count = len(included_messages)
        print(f"携带了 {history_count} 条历史信息")
        print(f"当前tokens: {total_tokens}")
        print(f"最大tokens: {tokensLimit}")

        return {"bot": baseModel, "message": message}
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to process messages. Error: {e}", "type": "error", "param": None, "code": 400}}, status_code=400) from e


async def generate_image(client: AsyncPoeApi, response: dict, aspect_ratio: str, image: list = []) -> str:
    try:
        async for chunk in client.send_message(bot=response["bot"], message=f"{response['message']} {aspect_ratio}", file_path=image):
            pass
        return chunk["text"]
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to generate image. Error: {e}", "type": "error", "param": None, "code": 500}}, status_code=500) from e
    
    
async def create_completion_data(
    completion_id: str, model: str, chunk: str = None, 
    finish_reason: str = None, include_usage: bool=False,
    prompt_tokens: int = 0, completion_tokens: int = 0 
) -> dict[str, Union[str, list, float]]:
    
    completion_timestamp = await helpers.__generate_timestamp()
    
    completion_data = ChatCompletionChunk(
        id=f"chatcmpl-{completion_id}",
        object="chat.completion.chunk",
        created=completion_timestamp,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=MessageResponse(role="assistant", content=chunk),
                finish_reason=finish_reason,
            )
        ],
    )
    
    if include_usage:
        completion_data.usage = None
        if finish_reason in ("stop", "length"):
            completion_data.usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens)
    
    return completion_data.dict()
    
    
async def generate_chunks(
    client: AsyncPoeApi, response: dict, model: str, completion_id: str, 
    prompt_tokens: int, image_urls: list[str], max_tokens: int, include_usage: bool,
    messages: list[dict[str, str]], enable_tracking: bool = False, client_id: str = None
) -> AsyncGenerator[bytes, None]:
    global remaining_balance, current_conversation_cost

    chat_id = response.get("chatId")
    chat_code = response.get("chatCode")
    last_chunk_text = ""  # 用于保存最后一个chunk的文本
    
    try:
        finish_reason = "stop"
        initial_balance = remaining_balance
        
        send_message_kwargs = {
            "bot": response["bot"],
            "message": response["message"],
            "file_path": image_urls
        }
        
        if chat_id and chat_code:
            send_message_kwargs.update({
                "chatId": chat_id,
                "chatCode": chat_code
            })
        
        async for chunk in client.send_message(**send_message_kwargs):
            chunk_token = await helpers.__tokenize(chunk["text"])
            
            if max_tokens and chunk_token >= max_tokens:
                await client.cancel_message(chunk)
                finish_reason = "length"
                break
            
            content = await create_completion_data(
                completion_id=completion_id,
                model=model,
                chunk=chunk["response"],
                include_usage=include_usage
            )
            
            yield b"data: " + orjson.dumps(content) + b"\n\n"
            await asyncio.sleep(0.001)
            
            if not chat_id:  # 如果是新会话，保存chat_id
                chat_id = chunk["chatId"]
                chat_code = chunk.get("chatCode")
        
        if enable_tracking:
            last_chunk_text = chunk["text"]  # 获取最后一个chunk的文本
            await helpers.__handle_conversation_state(
                messages,
                last_chunk_text,  # 使用最后一个chunk的文本作为assistant的回复
                {"chatId": chat_id, "chatCode": chat_code},
                client_id  # 传入client_id
            )
        
        # 重新获取剩余额度
        settings = await client.get_settings()
        remaining_balance = settings["messagePointInfo"]["messagePointBalance"]
        current_conversation_cost = initial_balance - remaining_balance
        print(f"本次对话消耗积分: {current_conversation_cost}")
        print(f"当前剩余额度（回复后）: {remaining_balance}")

        end_completion_data = await create_completion_data(
            completion_id=completion_id,
            model=model,
            finish_reason=finish_reason,
            include_usage=include_usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=chunk_token
        )
        
        yield b"data: " + orjson.dumps(end_completion_data) + b"\n\n"
        yield b"data: [DONE]\n\n"

        if chat_id is not None:
            try:
                await client.set_context_optimization(chat_id=chat_id, enabled=False)
                print(f"已关闭对话{chat_id}的自动管理上下文")
            except Exception as e:
                print(f"关闭自动管理上下文失败: {e}")
    except GeneratorExit:
        pass
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to stream response. Error: {e}", "type": "error", "param": None, "code": 500}}, status_code=500) from e

    
async def streaming_response(
    client: AsyncPoeApi, response: dict, model: str, completion_id: str, 
    prompt_tokens: int, image_urls: list[str], max_tokens: int, include_usage: bool, 
    text_messages: list, enable_tracking: bool, client_id: str
) -> StreamingResponse:
    
    return StreamingResponse(content=generate_chunks(client, response, model, completion_id, prompt_tokens, image_urls, max_tokens, include_usage, text_messages, enable_tracking, client_id), status_code=200, 
                             headers={"X-Request-ID": str(uuid.uuid4()), "Content-Type": "text/event-stream"})


async def non_streaming_response(
    client: AsyncPoeApi, response: dict, model: str, completion_id: str,
    prompt_tokens: int, image_urls: list[str], max_tokens: int
) -> ORJSONResponse:
    
    try:
        finish_reason = "stop"
        async for chunk in client.send_message(bot=response["bot"], message=response["message"], file_path=image_urls):
            if max_tokens and await helpers.__tokenize(chunk["text"]) >= max_tokens:
                await client.cancel_message(chunk)
                finish_reason = "length"
                break
            pass
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to generate completion. Error: {e}", "type": "error", "param": None, "code": 500}}, status_code=500) from e
    
    completion_tokens = await helpers.__tokenize(chunk["text"])
    
    content = ChatCompletionResponse(
        id=f"chatcmpl-{completion_id}",
        object="chat.completion",
        created=await helpers.__generate_timestamp(),
        model=model,
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=MessageResponse(role="assistant", content=chunk["text"]),
                finish_reason=finish_reason,
            )
        ],
    )
    
    return ORJSONResponse(content.dict())


async def rotate_token(tokens) -> tuple[AsyncPoeApi, bool, str]:
    global remaining_balance  # 声明使用全局变量
    if len(tokens) == 0:
        raise HTTPException(detail={"error": {"message": "All tokens have been used. Please add more tokens.", "type": "error", "param": None, "code": 402}}, status_code=402)
    token = random.choice(tokens)
    client = await AsyncPoeApi(token).create()
    settings = await client.get_settings()
    with open("data/settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    remaining_balance = settings["messagePointInfo"]["messagePointBalance"]
    print(f"当前剩余额度（回复前）: {remaining_balance}")
    if settings["messagePointInfo"]["messagePointBalance"] <= 20:
        tokens.remove(token)
        return await rotate_token(tokens)
    subscriptions = settings["subscription"]["isActive"]
    client_id = settings["subscription"]["id"]
    return client, subscriptions, client_id


if __name__ == "__main__":
    CommandLineInterface().run(["api:app", "--bind", "127.0.0.1", "--port", "8000"])
    
    
def start_server(tokens: list, address: str="127.0.0.1", port: str="8000"):
    if not isinstance(tokens, list):
        raise TypeError("Tokens must be a list.")
    if not all(isinstance(token, dict) for token in tokens):
        raise TypeError("Tokens must be a list of dictionaries.")
    app.state.tokens = tokens
    CommandLineInterface().run(["poe_api_wrapper.openai.api:app", "--bind", f"{address}", "--port", f"{port}"])