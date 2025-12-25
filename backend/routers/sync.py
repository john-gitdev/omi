import io
import os
import re
import struct
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Dict
from collections import defaultdict

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Header, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from opuslib import Decoder
from pydub import AudioSegment

from database import conversations as conversations_db
from database import users as users_db
from database.conversations import get_closest_conversation_to_timestamps, update_conversation_segments
from models.conversation import CreateConversation, ConversationSource, Conversation
from models.transcript_segment import TranscriptSegment
from utils.conversations.process_conversation import process_conversation
from utils.other import endpoints as auth
from utils.other.storage import (
    get_syncing_file_temporal_signed_url,
    delete_syncing_temporal_file,
    download_audio_chunks_and_merge,
    get_or_create_merged_audio,
    get_merged_audio_signed_url,
)
from utils import encryption
from utils.stt.pre_recorded import deepgram_prerecorded, postprocess_words
from utils.stt.vad import vad_is_empty

router = APIRouter()

# **********************************************
# ************ USER LOCK MANAGER ***************
# **********************************************

class UserLockManager:
    """
    Manages locks per user to prevent race conditions during DB updates.
    Uses reference counting to prevent memory leaks.
    """
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._active_batches: Dict[str, int] = defaultdict(int)
        self._global_lock = threading.Lock()

    def get_lock(self, uid: str) -> threading.Lock:
        """
        Get the lock object for a user and increment the active batch count.
        Does NOT acquire the lock.
        """
        with self._global_lock:
            if uid not in self._locks:
                self._locks[uid] = threading.Lock()
            self._active_batches[uid] += 1
            return self._locks[uid]

    def release_usage(self, uid: str):
        """
        Decrement active batch count and clean up lock if unused.
        """
        with self._global_lock:
            self._active_batches[uid] -= 1
            if self._active_batches[uid] <= 0:
                self._locks.pop(uid, None)
                self._active_batches.pop(uid, None)

user_locks = UserLockManager()


# **********************************************
# ******** AUDIO FORMAT CONVERSION *************
# **********************************************


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert PCM16 data to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """
    Parse HTTP Range header and return (start, end) tuple.
    Returns None if the range is invalid.
    """
    if not range_header:
        return None

    try:
        if not range_header.startswith("bytes="):
            return None

        range_spec = range_header[6:]
        parts = range_spec.split("-")

        if len(parts) != 2:
            return None

        start_str, end_str = parts

        if start_str and not end_str:
            start = int(start_str)
            end = file_size - 1
        elif not start_str and end_str:
            suffix_length = int(end_str)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str)

        if start < 0 or start >= file_size or start > end:
            return None
        end = min(end, file_size - 1)
        return (start, end)
    except (ValueError, IndexError):
        return None


# **********************************************
# ********** AUDIO PRE-CACHING *****************
# **********************************************


def _precache_audio_file(uid: str, conversation_id: str, audio_file: dict):
    """Pre-cache a single audio file."""
    try:
        audio_file_id = audio_file.get('id')
        timestamps = audio_file.get('chunk_timestamps')
        if not audio_file_id or not timestamps:
            return

        get_or_create_merged_audio(
            uid=uid,
            conversation_id=conversation_id,
            audio_file_id=audio_file_id,
            timestamps=timestamps,
            pcm_to_wav_func=pcm_to_wav,
        )
        print(f"Pre-cached audio file: {audio_file_id}")
    except Exception as e:
        print(f"Error pre-caching audio file {audio_file.get('id')}: {e}")


@router.post("/v1/sync/audio/{conversation_id}/precache", tags=['v1'])
def precache_conversation_audio_endpoint(
    conversation_id: str,
    uid: str = Depends(auth.get_current_user_uid),
):
    """
    Warm the audio cache for a conversation.
    """
    conversation = conversations_db.get_conversation(uid, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    audio_files = conversation.get('audio_files', [])
    if not audio_files:
        return {"status": "no_audio", "message": "No audio files in conversation"}

    def _precache_all_parallel():
        print(f"Pre-caching all {len(audio_files)} audio files for conversation {conversation_id} (parallel)")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_precache_audio_file, uid, conversation_id, af) for af in audio_files]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in parallel precache: {e}")
        print(f"Completed pre-cache for conversation {conversation_id}")

    thread = threading.Thread(target=_precache_all_parallel, daemon=True)
    thread.start()

    return {"status": "started", "audio_file_count": len(audio_files)}


@router.get("/v1/sync/audio/{conversation_id}/urls", tags=['v1'])
def get_audio_signed_urls_endpoint(
    conversation_id: str,
    uid: str = Depends(auth.get_current_user_uid),
):
    """
    Get signed URLs for all audio files in a conversation.
    """
    conversation = conversations_db.get_conversation(uid, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    audio_files = conversation.get('audio_files', [])
    if not audio_files:
        return {"audio_files": []}

    result = []
    uncached_files = []
    first_uncached_handled = False

    for af in audio_files:
        audio_file_id = af.get('id')
        if not audio_file_id:
            continue

        signed_url = get_merged_audio_signed_url(uid, conversation_id, audio_file_id)

        if signed_url:
            result.append(
                {
                    "id": audio_file_id,
                    "status": "cached",
                    "signed_url": signed_url,
                    "duration": af.get('duration', 0),
                }
            )
        else:
            if not first_uncached_handled:
                first_uncached_handled = True
                _precache_audio_file(uid, conversation_id, af)
                signed_url = get_merged_audio_signed_url(uid, conversation_id, audio_file_id)
                if signed_url:
                    result.append(
                        {
                            "id": audio_file_id,
                            "status": "cached",
                            "signed_url": signed_url,
                            "duration": af.get('duration', 0),
                        }
                    )
                else:
                    result.append(
                        {
                            "id": audio_file_id,
                            "status": "pending",
                            "signed_url": None,
                            "duration": af.get('duration', 0),
                        }
                    )
            else:
                result.append(
                    {
                        "id": audio_file_id,
                        "status": "pending",
                        "signed_url": None,
                        "duration": af.get('duration', 0),
                    }
                )
                uncached_files.append(af)

    if uncached_files:
        def _cache_uncached_parallel():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_precache_audio_file, uid, conversation_id, af) for af in uncached_files]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in parallel cache: {e}")

        thread = threading.Thread(target=_cache_uncached_parallel, daemon=True)
        thread.start()

    return {"audio_files": result}


# **********************************************
# ********** AUDIO DOWNLOAD ENDPOINT ***********
# **********************************************


@router.get("/v1/sync/audio/{conversation_id}/{audio_file_id}", tags=['v1'])
def download_audio_file_endpoint(
    conversation_id: str,
    audio_file_id: str,
    request: Request,
    format: str = Query(default="wav", regex="^(wav|pcm)$"),
    uid: str = Depends(auth.get_current_user_uid),
):
    """
    Download audio file from private cloud sync in the specified format.
    """
    conversation = conversations_db.get_conversation(uid, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    audio_files = conversation.get('audio_files', [])
    audio_file = None
    for af in audio_files:
        if af.get('id') == audio_file_id:
            audio_file = af
            break

    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found in conversation")

    try:
        if not audio_file.get('chunk_timestamps'):
            raise HTTPException(status_code=500, detail="Audio file has no chunk timestamps")

        if format == "wav":
            audio_data, was_cached = get_or_create_merged_audio(
                uid=uid,
                conversation_id=conversation_id,
                audio_file_id=audio_file_id,
                timestamps=audio_file['chunk_timestamps'],
                pcm_to_wav_func=pcm_to_wav,
            )
            content_type = "audio/wav"
            extension = "wav"
        else:
            audio_data = download_audio_chunks_and_merge(uid, conversation_id, audio_file['chunk_timestamps'])
            content_type = "application/octet-stream"
            extension = "pcm"
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio chunks not found in storage")
    except Exception as e:
        print(f"Error downloading audio file: {e}")
        raise HTTPException(status_code=500, detail="Failed to download audio file")

    filename = f"conversation_{conversation_id}_audio_{audio_file_id}.{extension}"
    file_size = len(audio_data)

    base_headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }

    range_header = request.headers.get("Range")

    if range_header:
        range_tuple = parse_range_header(range_header, file_size)

        if range_tuple is None:
            return Response(
                status_code=416,
                headers={
                    "Content-Range": f"bytes */{file_size}",
                    **base_headers,
                },
            )

        start, end = range_tuple
        content_length = end - start + 1

        return StreamingResponse(
            io.BytesIO(audio_data[start : end + 1]),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Length": str(content_length),
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                **base_headers,
            },
        )

    return StreamingResponse(
        io.BytesIO(audio_data),
        status_code=200,
        media_type=content_type,
        headers={
            "Content-Length": str(file_size),
            **base_headers,
        },
    )


# **********************************************
# ************ SYNC LOCAL FILES ****************
# **********************************************


import shutil
import wave


def decode_opus_file_to_wav(opus_file_path, wav_file_path, sample_rate=16000, channels=1, frame_size: int = 160):
    """Decode an Opus file with length-prefixed frames to WAV format."""
    if not os.path.exists(opus_file_path):
        print(f"File not found: {opus_file_path}")
        return False

    decoder = Decoder(sample_rate, channels)
    with open(opus_file_path, 'rb') as f:
        pcm_data = []
        frame_count = 0
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                print("End of file reached.")
                break
            if len(length_bytes) < 4:
                print("Incomplete length prefix at the end of the file.")
                break

            frame_length = struct.unpack('<I', length_bytes)[0]
            opus_data = f.read(frame_length)
            if len(opus_data) < frame_length:
                print(f"Unexpected end of file at frame {frame_count}.")
                break
            try:
                pcm_frame = decoder.decode(opus_data, frame_size=frame_size)
                pcm_data.append(pcm_frame)
                frame_count += 1
            except Exception as e:
                print(f"Error decoding frame {frame_count}: {e}")
                break
        if pcm_data:
            pcm_bytes = b''.join(pcm_data)
            with wave.open(wav_file_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)
            print(f"Decoded audio saved to {wav_file_path}")
            return True
        else:
            print("No PCM data was decoded.")
            return False


def get_timestamp_from_path(path: str):
    timestamp = int(path.split('/')[-1].split('_')[-1].split('.')[0])
    if timestamp > 1e10:
        return int(timestamp / 1000)
    return timestamp


def retrieve_file_paths(files: List[UploadFile], uid: str):
    directory = f'syncing/{uid}/'
    os.makedirs(directory, exist_ok=True)
    paths = []
    for file in files:
        filename = file.filename
        if not filename.endswith('.bin'):
            raise HTTPException(status_code=400, detail=f"Invalid file format {filename}")
        if '_' not in filename:
            raise HTTPException(status_code=400, detail=f"Invalid file format {filename}, missing timestamp")
        try:
            timestamp = get_timestamp_from_path(filename)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid file format {filename}, invalid timestamp")

        time = datetime.fromtimestamp(timestamp)
        if time > datetime.now() or time < datetime(2024, 1, 1):
            raise HTTPException(status_code=400, detail=f"Invalid file format {filename}, invalid timestamp")

        path = f"{directory}{filename}"
        try:
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            paths.append(path)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise HTTPException(status_code=500, detail=f"Failed to write file {filename}: {str(e)}")
    return paths


def decode_files_to_wav(files_path: List[str]):
    wav_files = []
    for path in files_path:
        wav_path = path.replace('.bin', '.wav')
        filename = os.path.basename(path)
        frame_size = 160  # Default frame size
        match = re.search(r'_fs(\d+)', filename)
        if match:
            try:
                frame_size = int(match.group(1))
                print(f"Found frame size {frame_size} in filename: {filename}")
            except ValueError:
                print(f"Invalid frame size format in filename: {filename}, using default {frame_size}")

        success = decode_opus_file_to_wav(path, wav_path, frame_size=frame_size)
        if not success:
            continue

        try:
            aseg = AudioSegment.from_wav(wav_path)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=400, detail=f"Invalid file format {path}, {e}")

        if aseg.duration_seconds < 1:
            os.remove(wav_path)
            continue
        wav_files.append(wav_path)
        if os.path.exists(path):
            os.remove(path)
    return wav_files


def retrieve_vad_segments(path: str, segmented_paths: set):
    start_timestamp = get_timestamp_from_path(path)
    voice_segments = vad_is_empty(path, return_segments=True, cache=True)

    segments = []
    for i, segment in enumerate(voice_segments):
        if segments and (segment['start'] - segments[-1]['end']) < 120:
            segments[-1]['end'] = segment['end']
        else:
            segments.append(segment)

    print(path, len(segments))

    aseg = AudioSegment.from_wav(path)
    path_dir = '/'.join(path.split('/')[:-1])
    for i, segment in enumerate(segments):
        if (segment['end'] - segment['start']) < 1:
            continue
        segment_timestamp = start_timestamp + segment['start']
        segment_path = f'{path_dir}/{segment_timestamp}.wav'
        segment_aseg = aseg[segment['start'] * 1000 : segment['end'] * 1000]
        segment_aseg.export(segment_path, format='wav')
        segmented_paths.add(segment_path)


def _reprocess_conversation_after_update(uid: str, conversation_id: str, language: str):
    """
    Reprocess a conversation after new segments have been added.
    """
    conversation_data = conversations_db.get_conversation(uid, conversation_id)
    if not conversation_data:
        print(f'Conversation {conversation_id} not found for reprocessing')
        return

    conversation = Conversation(**conversation_data)

    process_conversation(
        uid=uid,
        language_code=language or 'en',
        conversation=conversation,
        force_process=True,
        is_reprocess=True,
    )

    print(f'Successfully reprocessed conversation {conversation_id}')


def process_segment(path: str, uid: str, response: dict, memory_languages: dict, db_lock: threading.Lock, source: ConversationSource = ConversationSource.omi):
    try:
        # Get the URL for Deepgram
        url = get_syncing_file_temporal_signed_url(path)

        # 1. Transcribe (Slow, safe to run in parallel)
        # Note: We removed the delete_file thread. Cleanup is now handled in 'finally'.
        words, language = deepgram_prerecorded(url, speakers_count=3, attempts=0, return_language=True)
        transcript_segments: List[TranscriptSegment] = postprocess_words(words, 0)
        
        if not transcript_segments:
            print(f'Failed to get deepgram segments for {path}')
            return

        timestamp = get_timestamp_from_path(path)
        segment_end_timestamp = timestamp + transcript_segments[-1].end

        # 2. Critical Section (Fast, must be sequential)
        with db_lock:
            closest_memory = get_closest_conversation_to_timestamps(uid, timestamp, segment_end_timestamp)

            if closest_memory:
                memory_languages[closest_memory['id']] = language

            if not closest_memory:
                started_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                finished_at = datetime.fromtimestamp(segment_end_timestamp, tz=timezone.utc)
                create_memory = CreateConversation(
                    started_at=started_at,
                    finished_at=finished_at,
                    transcript_segments=transcript_segments,
                    source=source,
                )
                created = process_conversation(uid, language, create_memory)
                response['new_memories'].add(created.id)
            else:
                transcript_segments = [s.dict() for s in transcript_segments]
                for segment in transcript_segments:
                    segment['timestamp'] = timestamp + segment['start']
                for segment in closest_memory['transcript_segments']:
                    segment['timestamp'] = closest_memory['started_at'].timestamp() + segment['start']

                segments = closest_memory['transcript_segments'] + transcript_segments
                segments.sort(key=lambda x: x['timestamp'])

                for i, segment in enumerate(segments):
                    duration = segment['end'] - segment['start']
                    segment['start'] = segment['timestamp'] - closest_memory['started_at'].timestamp()
                    segment['end'] = segment['start'] + duration

                last_segment_end = segments[-1]['end'] if segments else 0
                new_finished_at = datetime.fromtimestamp(
                    closest_memory['started_at'].timestamp() + last_segment_end, tz=timezone.utc
                )

                if new_finished_at < closest_memory['finished_at']:
                    new_finished_at = closest_memory['finished_at']

                for segment in segments:
                    segment.pop('timestamp')

                response['updated_memories'].add(closest_memory['id'])
                update_conversation_segments(uid, closest_memory['id'], segments, finished_at=new_finished_at)

    except Exception as e:
        print(f"Error processing segment {path}: {e}")
    finally:
        # 3. Explicit Cleanup (Always runs)
        # We delete the file only AFTER we are completely done with it.
        try:
            delete_syncing_temporal_file(path)
        except Exception as e:
            print(f"Failed to cleanup file {path}: {e}")

def process_audio_uploads_background(paths: List[str], uid: str, source: ConversationSource):
    print(f"Background processing started for {len(paths)} files for user {uid}")
    
    wav_paths = decode_files_to_wav(paths)

    def chunk_threads(threads):
        chunk_size = 5
        for i in range(0, len(threads), chunk_size):
            [t.start() for t in threads[i : i + chunk_size]]
            [t.join() for t in threads[i : i + chunk_size]]

    segmented_paths = set()
    threads = [threading.Thread(target=retrieve_vad_segments, args=(path, segmented_paths)) for path in wav_paths]
    chunk_threads(threads)

    print('Background VAD complete. Segments:', len(segmented_paths))

    response = {'updated_memories': set(), 'new_memories': set()}
    memory_languages = {}

    # 1. Get lock object (but don't acquire it yet)
    # This prevents memory leak by counting usage
    user_lock = user_locks.get_lock(uid)
    
    try:
        threads = [
            threading.Thread(
                target=process_segment,
                args=(path, uid, response, memory_languages, user_lock, source),
            )
            for path in segmented_paths
        ]
        chunk_threads(threads)
        
        # 4. Reprocess Updated Memories
        if response['updated_memories']:
            print(f"Reprocessing {len(response['updated_memories'])} updated conversations...")
            # We lock here too to ensure we don't reprocess while a new batch is writing
            with user_lock:
                for memory_id in response['updated_memories']:
                    try:
                        detected_language = memory_languages.get(memory_id, 'en') 
                        print(f"Reprocessing memory {memory_id} in {detected_language}")
                        _reprocess_conversation_after_update(uid, memory_id, language=detected_language)
                    except Exception as e:
                        print(f"Failed to reprocess memory {memory_id}: {e}")
                        
    finally:
        # 2. Release usage count so lock can be cleaned up
        user_locks.release_usage(uid)

    print(f"Background processing complete. Updated: {response['updated_memories']}, New: {response['new_memories']}")

@router.post("/v1/sync-local-files")
async def sync_local_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...), 
    uid: str = Depends(auth.get_current_user_uid)
):
    source = ConversationSource.omi
    for f in files:
        if f.filename and 'limitless' in f.filename.lower():
            source = ConversationSource.limitless
            break

    paths = retrieve_file_paths(files, uid)
    background_tasks.add_task(process_audio_uploads_background, paths, uid, source)

    return {
        "status": "processing", 
        "message": "Uploads accepted, processing in background",
        "new_memories": [],      
        "updated_memories": []   
    }