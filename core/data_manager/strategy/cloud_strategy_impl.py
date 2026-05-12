import asyncio
import json
import logging
import os
import time
import uuid
import re
import base64
import tempfile
import numpy as np
from typing import List, Dict, Optional, Any, Set
from datetime import date

from core.data_manager.strategy.base_strategy import StorageStrategy, SessionContext

# Cloud SDK imports
from wt_sdk import WTGatewayClient, GatewayConfig, EnvConfigManager
from wt_sdk.models import LandingRecord, ChatMessage, ContentItem
from wt_sdk.utils import generate_deterministic_id, S3Uploader, S3Downloader

log = logging.getLogger("cloud_strategy")

# Retry configuration
MAX_UPLOAD_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0

class CloudStrategy(StorageStrategy):
    """
    Cloud storage strategy:
    - Table 1 (S3): Environment configs stored via EnvConfigManager
    - Table 2 (LandingTable): Session steps with full conversation history

    Image handling:
    - Extract base64 images from messages
    - Upload binary to S3 with retry logic
    - On failure: store locally as fallback
    - Store S3 URLs (or local paths) in messages JSON
    """

    def __init__(
        self,
        job_id: str,
        db_url: str,
        enable_buffer: bool = False,
        buffer_size: int = 1,
        flush_interval: float = 1.0
    ):
        self.db_url = db_url
        self.job_id = job_id
        self.initialized = False

        self.client: Optional[WTGatewayClient] = None
        self.env_manager: Optional[EnvConfigManager] = None
        self.s3_uploader: Optional[S3Uploader] = None

        self._enable_buffer = enable_buffer
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._record_buffer: List[LandingRecord] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._stats = {
            "total_create_buffered": 0,
            "total_create_flushed": 0,
            "flush_count": 0,
        }

        # In-memory caches
        self._env_configs: Dict[str, Dict] = {}
        self._sessions: Dict[str, SessionContext] = {}

        # Local fallback directory for failed uploads
        self._local_fallback_dir = "saved_images"

    async def init(self):
        """Initialize cloud clients"""
        if self.initialized:
            return

        # TODO After the test passed, it was modified to the production configuration.
        # 1. Initialize WTGatewayClient
        config = GatewayConfig() 
        config.tables.landing_table = "landing_test"

        try:
            self.client = WTGatewayClient(config)
            log.info(f"CloudStrategy initialized with table: {config.tables.landing_table}")
        except Exception as e:
            log.error(f"Failed to initialize WTGatewayClient: {e}")
            raise

        # 2. Initialize EnvConfigManager (S3)
        try:
            self.env_manager = EnvConfigManager() 
        except Exception as e:
            log.error(f"Failed to initialize EnvConfigManager: {e}")
            raise
        
        # 3. Initialize S3Uploader
        try:
            self.s3_uploader = S3Uploader()
        except Exception as e:
            log.warning(f"Failed to initialize S3Uploader: {e}")
            self.s3_uploader = None

        self.initialized = True

        if self._enable_buffer and not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._periodic_flush())
            log.info(
                "Cloud buffer started: buffer_size=%d flush_interval=%.2fs",
                self._buffer_size,
                self._flush_interval,
            )

    async def add_environment(
        self,
        job_id: str,
        env_name: str,
        env_params: Dict,
        image: str = "",
        group_id: str = ""
    ) -> str:
        """Register environment config to S3"""
        await self.init()
        
        env_id = str(uuid.uuid4())
        
        config_dict = {
            "job_id": job_id,
            "env_id": env_id,
            "env_name": env_name,
            "env_params": env_params,
            "image": image,
            "group_id": group_id,
            "created_at": int(time.time()),
        }
        
        try:
            await asyncio.to_thread(self.env_manager.save_config, config_dict)
            log.info(f"Environment config saved to S3: {env_id}")
        except Exception as e:
            log.error(f"Failed to save env config to S3: {e}")
            
        # Cache locally
        self._env_configs[env_id] = config_dict

        return env_id

    async def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict]:
        """Get all environments from cache"""
        if job_id:
            return [c for c in self._env_configs.values() if c.get("job_id") == job_id]
        return list(self._env_configs.values())

    async def create_session(
        self,
        env_id: str,
        env_name: str,
        llm_model: str,
        group_id: str = "",
        job_id: str = ""
    ) -> SessionContext:
        """Create session context (in-memory only)"""
        session = SessionContext(
            session_id=env_id,
            env_id=env_id,
            env_name=env_name,
            llm_model=llm_model,
            group_id=group_id,
            job_id=job_id or self.job_id,
            total_reward=0.0,
            start_time=time.perf_counter(),
            message_history=[]
        )

        self._sessions[session.session_id] = session
        log.debug("Created cloud session: %s", session.session_id)
        return session

    async def record_step(
        self,
        session: SessionContext,
        step_id: int,
        messages: List[Dict],
        response: str,
        step_reward: float,
        env_state: Optional[str] = None,
        terminated: bool = False,
        truncated: bool = False,
        is_trainable: bool = True
    ):
        """
        Record step to cloud LandingTable.
        Images are extracted, uploaded to S3 (with retry), and URLs stored.
        """
        await self.init()
        
        session.total_reward += step_reward

        env_key = f"{session.env_name}_{session.env_id}"

        # Optimization: session.message_history already holds previously processed
        # messages with S3 URLs substituted for base64.  Reuse that prefix and only
        # process images in the *new* messages appended since the last step, avoiding
        # redundant re-uploads of the same images on every cumulative call.
        prev_count = len(session.message_history)
        if prev_count > 0 and len(messages) >= prev_count:
            new_messages = messages[prev_count:]
            new_processed, image_urls = await self._process_images(
                new_messages, env_key, step_id
            )
            full_messages = list(session.message_history) + list(new_processed)
        else:
            # First step or unexpected message count — process everything normally.
            full_messages, image_urls = await self._process_images(
                messages, env_key, step_id
            )

        # full_messages.append({"role": "assistant", "content": response})
        session.message_history = full_messages
        
        # Convert to SDK format
        chat_messages = self._convert_to_chat_messages(full_messages)
        response_msg = ChatMessage(
            role="assistant",
            content=[ContentItem(type="text", text=response)]
        )
        
        # Generate deterministic record ID
        record_id = generate_deterministic_id({
            "session_id": session.session_id,
            "step_id": step_id,
            "env_name": session.env_name
        })
        
        # Create LandingRecord
        record = LandingRecord(
            dataset_type="Test",
            dt=date.today().isoformat(),
            id=record_id,
            session_id=session.session_id,
            step_id=step_id,
            env_id=session.env_id,
            job_id=session.job_id,
            created_at=int(time.time()),
            step_reward=step_reward,
            reward=session.total_reward,
            messages=chat_messages,
            response=response_msg,
            ground_truth_answer=None,
            reference_answer=None,
            agent_model=session.llm_model,
            env_name=session.env_name,
            is_terminal=terminated,
            is_truncated=truncated,
            is_session_completed=terminated or truncated,
            meta_json=json.dumps({
                "source": "AIEvoBox",
                "group_id": session.group_id,
                "env_state": env_state,
                "is_trainable": bool(is_trainable),
            })
        )
        
        if self._enable_buffer:
            await self._buffer_record(record)
        else:
            try:
                await asyncio.to_thread(self.client.ingest_landing, record)
                log.debug("Step %d recorded to cloud: %s", step_id, record_id)
            except Exception as e:
                log.error("Failed to ingest step %d: %s", step_id, e)
                raise

    async def close(self) -> None:
        """Clean up cloud clients"""
        if self._enable_buffer:
            await self._stop_buffer()

        if self.client and hasattr(self.client, 'close'):
            self.client.close()

        if self.env_manager and hasattr(self.env_manager, 'close'):
            self.env_manager.close()

        self.initialized = False
        log.info("Cloud strategy closed")
                
    def get_sync_connection(self) -> None:
        """Not applicable for cloud storage"""
        return None

    @property
    def buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics"""
        return self._stats if self._enable_buffer else None
    
    async def _process_images(
        self,
        messages: List[Dict],
        env_key: str,
        step_id: int
    ) -> tuple[List[Dict], List[str]]:
        """
        Process images in messages:
        1. Extract base64 images
        2. Upload to S3 with retry
        3. On failure: save locally as fallback
        4. Replace base64 with URL/path in messages
        """
        processed_messages = []
        uploaded_urls = []
        image_count = 0

        for msg_idx, message in enumerate(messages):
            content = message.get("content")

            # Skip non-list content or content without images
            if not isinstance(content, list):
                processed_messages.append(message)
                continue

            has_images = any(item.get("type") == "image_url" for item in content)
            if not has_images:
                processed_messages.append(message)
                continue

            # Process images in content
            new_message = message.copy()
            new_content = []

            for item_idx, item in enumerate(content):
                if item.get("type") != "image_url":
                    new_content.append(item)
                    continue

                image_url = item.get("image_url", {}).get("url", "")

                # Check if base64
                match = re.match(r"data:image/(\w+);base64,(.+)", image_url)
                if not match:
                    new_content.append(item)
                    continue

                # Extract image data
                ext = match.group(1)
                b64_str = match.group(2)
                file_name = f"step_{step_id}_m{msg_idx}_i{image_count}.{ext}"

                # Upload with retry
                final_url = await self._upload_image_with_retry(
                    b64_str, env_key, file_name, ext
                )

                # Update item
                new_item = item.copy()
                new_item["image_url"] = item["image_url"].copy()
                new_item["image_url"]["url"] = final_url

                new_content.append(new_item)
                uploaded_urls.append(final_url)
                image_count += 1

            new_message["content"] = new_content
            processed_messages.append(new_message)

        return processed_messages, uploaded_urls

    async def _buffer_record(self, record: LandingRecord) -> None:
        should_flush = False

        async with self._buffer_lock:
            self._record_buffer.append(record)
            self._stats["total_create_buffered"] += 1
            if len(self._record_buffer) >= self._buffer_size:
                should_flush = True

        if should_flush:
            self._create_flush_task(self._flush_records())

    def _create_flush_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _periodic_flush(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._flush_interval)
                await self._flush_records()
        except asyncio.CancelledError:
            raise

    async def _stop_buffer(self) -> None:
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self._flush_records()
        log.info("Cloud buffer stopped: stats=%s", self._stats)

    async def _flush_records(self) -> int:
        async with self._flush_lock:
            async with self._buffer_lock:
                if not self._record_buffer:
                    return 0
                records = list(self._record_buffer)
                self._record_buffer.clear()

            try:
                await asyncio.to_thread(self.client.ingest_landing_batch, records)
            except Exception as e:
                async with self._buffer_lock:
                    self._record_buffer = records + self._record_buffer
                log.error("Failed to flush %d cloud records: %s", len(records), e)
                raise

            self._stats["total_create_flushed"] += len(records)
            self._stats["flush_count"] += 1
            log.debug("Flushed %d cloud records", len(records))
            return len(records)
    
    async def _upload_image_with_retry(
        self,
        b64_str: str,
        env_key: str,
        file_name: str,
        ext: str
    ) -> str:
        """
        Upload image to S3 with retry logic.
        Falls back to local storage on failure.
        """
        # Decode base64
        try:
            img_data = base64.b64decode(b64_str)
        except Exception as e:
            log.error("Failed to decode base64: %s", e)
            return f"data:image/{ext};base64,{b64_str[:50]}..."  # Keep partial for debugging

        # Create local fallback path
        local_dir = os.path.join(self._local_fallback_dir, env_key)
        local_path = os.path.join(local_dir, file_name)

        # Try S3 upload with retry
        if self.s3_uploader:
            s3_key = f"aievobox/{self.job_id}/{env_key}/{file_name}"

            for attempt in range(MAX_UPLOAD_RETRIES):
                try:
                    # Save to temp file first
                    os.makedirs(local_dir, exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(img_data)

                    # Upload to S3
                    s3_url = await asyncio.to_thread(
                        self.s3_uploader.upload_file,
                        file_path=local_path,
                        key=s3_key
                    )

                    if s3_url:
                        log.debug("Uploaded image to S3: %s", s3_url)
                        return s3_url

                except Exception as e:
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    log.warning(
                        "S3 upload failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_UPLOAD_RETRIES, e, wait_time
                    )
                    await asyncio.sleep(wait_time)

            log.error("S3 upload failed after %d retries. Using local fallback.", MAX_UPLOAD_RETRIES)

        # Fallback to local storage
        try:
            os.makedirs(local_dir, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(img_data)
            log.info("Image saved locally: %s", local_path)
            return local_path
        except Exception as e:
            log.error("Local save also failed: %s", e)
            return f"[IMAGE_SAVE_FAILED:{file_name}]"
        
    async def fetch_done_steps_with_context(
        self,
        job_id: str,
        after_id: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch trainable steps for training data collection.
        Uses cursor-based pagination.
        """
        await self.init()
        
        results = self.client.pull_data(
            dataset_type="Test",
            cursor=after_id,
            checkout_latest=True,
            where_sql="job_id = '{}'".format(job_id),
            limit=limit,
        )
        
        if results is None or len(results) == 0:
            print(f"len results {len(results)} No more data to fetch")
            return []
        
        cursor = self.client.extract_cursor(results)
        
        rows = []
        for _, row in results.iterrows():
            meta_json = json.loads(row["meta_json"]) if row["meta_json"] else {}
            if meta_json.get("is_trainable") is False:
                continue
            rows.append(
                {
                    "step_pk": cursor,
                    "step_id": row["step_id"],
                    "env_name": row["env_name"],
                    "env_id": row["session_id"],
                    "env_state": meta_json.get("env_state"),
                    "prompt": self.normalize_messages(row["messages"]),
                    "response": row["response"]["content"].tolist()[0]["text"],
                    "reward": row["reward"],
                    "step_reward": row["step_reward"],
                    "total_reward": row["reward"],
                    "session_id": row["session_id"],
                    "session_end_time": row["created_at"] if row["created_at"] else None,
                    "group_id": meta_json.get("group_id"),
                    "truncated": row["is_truncated"],
                    "is_session_completed": row["is_session_completed"], 
                }
            )
        return rows
        
    async def get_max_step_id(self, job_id: str) -> int:
        """Get maximum primary key for pagination"""
        await self.init()
        
        last_cursor = self.client.get_max_created_at(
            where_sql="dataset_type = 'Test' AND job_id = '{}'".format(job_id),
        )
        
        return last_cursor

    # --- Helpers ---
    def ndarray_to_native(self, obj: Any) -> Any:
        """
        Recursively remove numpy.array / numpy scalar and convert to native Python types
        """
        
        if isinstance(obj, np.ndarray):
            return [self.ndarray_to_native(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, list):
            return [self.ndarray_to_native(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.ndarray_to_native(v) for k, v in obj.items()}
        return obj
    
    def extract_image_path(self, item: dict) -> str | None:
        """
        Extract image path:
        1. Priority: item["image_url"]["url"]
        2. Otherwise, try s3:// or http(s):// within item["text"]
        """
        
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if url:
                return url

        text = item.get("text")
        if isinstance(text, str) and text.startswith(("s3://", "http://", "https://")):
            return text

        return None
    
    def download_image_as_base64(self, image_path: str) -> tuple[str, str | None]:
        """
        Use S3Downloader to download the image and convert it to base64

        Returns:
            (base64_string, media_type)
        """
        downloader = S3Downloader()

        suffix = os.path.splitext(image_path)[1]
        if not suffix:
            suffix = ".bin"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, f"image{suffix}")
            downloader.download_file(image_path, local_path)

            with open(local_path, "rb") as f:
                image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        media_type = suffix[1:]
        return image_base64, media_type
    
    def remove_none_and_empty(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                v = self.remove_none_and_empty(v)
                if v is not None and v != [] and v != {}:
                    cleaned[k] = v
            return cleaned

        if isinstance(obj, list):
            return [self.remove_none_and_empty(x) for x in obj]

        return obj
    
    def normalize_messages(self, messages: Any) -> list:
        messages = self.ndarray_to_native(messages)

        if isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "image_url":
                    image_path = self.extract_image_path(item)
                    if image_path:
                        try:
                            image_base64, media_type = self.download_image_as_base64(image_path)
                            item.clear()
                            item["type"] = "image_url"
                            item["image_url"] = {
                                "url": f"data:image/{media_type};base64,{image_base64}"
                            }
                        except Exception as e:
                            item.clear()
                            item["type"] = "image_error"
                            item["error"] = str(e)
                            item["source"] = image_path

        messages = self.remove_none_and_empty(messages)
        return messages

    def _convert_to_chat_messages(self, messages: List[Dict]) -> List[ChatMessage]:
        """Convert OpenAI format messages to SDK ChatMessage format"""
        result = []

        for msg in messages:
            role = msg.get("role", "user")
            content_raw = msg.get("content", "")

            content_items = []
            if isinstance(content_raw, str):
                content_items.append(ContentItem(type="text", text=content_raw))
            elif isinstance(content_raw, list):
                for item in content_raw:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_items.append(
                                ContentItem(type="text", text=item.get("text", ""))
                            )
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            content_items.append(
                                ContentItem(type="image_url", text=url)
                            )

            result.append(ChatMessage(role=role, content=content_items))

        return result
