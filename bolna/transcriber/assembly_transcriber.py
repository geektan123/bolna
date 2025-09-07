import asyncio
import traceback
import os
import json
import aiohttp
import time
from urllib.parse import urlencode
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class AssemblyTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='best', stream=True, language="en", 
                 endpointing="400", sampling_rate="16000", encoding="linear16", output_queue=None, 
                 keywords=None, process_interim_results="true", **kwargs):
        super().__init__(input_queue)
        self.endpointing = endpointing
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = model
        self.sampling_rate = 16000
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('ASSEMBLYAI_API_KEY'))
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.keywords = keywords
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.process_interim_results = process_interim_results
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        
        # Message states
        self.curr_message = ''
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.websocket_connection = None
        self.connection_authenticated = False
        
        # Assembly AI specific settings
        self.base_url = "https://api.assemblyai.com"
        self.websocket_url = "wss://api.assemblyai.com/v2/realtime/ws"

    def get_assembly_ws_url(self):
        """Generate Assembly AI websocket URL with parameters"""
        params = {
            'sample_rate': self.sampling_rate,
            'word_boost': self.keywords if self.keywords else '',
            'encoding': self.encoding,
            'language_code': self.language,
            'endpointing': self.endpointing,
            'punctuate': 'true',
            'format_text': 'true',
            'interim_results': 'true'
        }
        
        # Adjust parameters based on telephony provider
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'mulaw' if self.provider == "twilio" else "linear16"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2
            params['sample_rate'] = self.sampling_rate
            params['encoding'] = self.encoding
        elif self.provider == "web_based_call":
            params['encoding'] = "linear16"
            params['sample_rate'] = 16000
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
        elif not self.connected_via_dashboard:
            params['encoding'] = "linear16"
            params['sample_rate'] = 16000
            
        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0
            
        # Build websocket URL
        websocket_url = f"{self.websocket_url}?{urlencode(params)}"
        return websocket_url

    async def send_heartbeat(self, ws: ClientConnection):
        """Send heartbeat to keep connection alive"""
        try:
            while True:
                data = {'type': 'ping'}
                try:
                    await ws.send(json.dumps(data))
                except ConnectionClosedError as e:
                    logger.info(f"Connection closed while sending heartbeat: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    break
                    
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error('Error in send_heartbeat: ' + str(e))
            raise

    async def toggle_connection(self):
        """Toggle connection state and cleanup resources"""
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        if self.sender_task is not None:
            self.sender_task.cancel()
        
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("Websocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing websocket connection: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _get_http_transcription(self, audio_data):
        """Get transcription via HTTP API (for non-streaming mode)"""
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'audio/wav'
        }

        self.current_request_id = self.generate_request_id()
        self.meta_info['request_id'] = self.current_request_id
        start_time = time.time()
        
        # Upload audio file first
        upload_url = f"{self.base_url}/v2/upload"
        async with self.session.post(upload_url, data=audio_data, headers=headers) as upload_response:
            if upload_response.status != 200:
                raise Exception(f"Failed to upload audio: {await upload_response.text()}")
            upload_data = await upload_response.json()
            audio_url = upload_data['upload_url']

        # Submit transcription job
        transcript_url = f"{self.base_url}/v2/transcript"
        transcript_data = {
            'audio_url': audio_url,
            'language_code': self.language,
            'punctuate': True,
            'format_text': True
        }
        
        async with self.session.post(transcript_url, json=transcript_data, headers=headers) as transcript_response:
            if transcript_response.status != 200:
                raise Exception(f"Failed to submit transcription: {await transcript_response.text()}")
            transcript_data = await transcript_response.json()
            transcript_id = transcript_data['id']

        # Poll for results
        while True:
            async with self.session.get(f"{self.base_url}/v2/transcript/{transcript_id}", headers=headers) as status_response:
                if status_response.status != 200:
                    raise Exception(f"Failed to get transcript status: {await status_response.text()}")
                status_data = await status_response.json()
                
                if status_data['status'] == 'completed':
                    self.meta_info["start_time"] = start_time
                    self.meta_info['transcriber_latency'] = time.time() - start_time
                    transcript = status_data['text']
                    self.meta_info['transcriber_duration'] = status_data.get('audio_duration', 0)
                    return create_ws_data_packet(transcript, self.meta_info)
                elif status_data['status'] == 'error':
                    raise Exception(f"Transcription failed: {status_data.get('error', 'Unknown error')}")
                
                await asyncio.sleep(1)  # Poll every second

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal"""
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            await self._close(ws, data={"type": "CloseStream"})
            return True
        return False

    def get_meta_info(self):
        """Get metadata information"""
        return self.meta_info

    async def sender(self, ws=None):
        """Sender for non-streaming mode"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if not self.audio_submitted:
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                self.meta_info = ws_data_packet.get('meta_info')
                start_time = time.time()
                transcription = await self._get_http_transcription(ws_data_packet.get('data'))
                transcription['meta_info']["include_latency"] = True
                transcription['meta_info']["transcriber_latency"] = time.time() - start_time
                transcription['meta_info']['audio_duration'] = transcription['meta_info']['transcriber_duration']
                transcription['meta_info']['last_vocal_frame_timestamp'] = start_time
                yield transcription

            if self.transcription_task is not None:
                self.transcription_task.cancel()
        except asyncio.CancelledError:
            logger.info("Cancelled sender task")
            return

    async def sender_stream(self, ws: ClientConnection):
        """Sender for streaming mode"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                # Initialize new request
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                self.num_frames += 1
                # Save the audio cursor here
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                
                try:
                    await ws.send(ws_data_packet.get('data'))
                except ConnectionClosedError as e:
                    logger.error(f"Connection closed while sending data: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending data to websocket: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error('Error in sender_stream: ' + str(e))
            raise

    async def receiver(self, ws: ClientConnection):
        """Receiver for streaming responses"""
        async for msg in ws:
            try:
                msg = json.loads(msg)

                # If connection_start_time is None, it is the durations of frame submitted till now minus current time
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                if msg.get("message_type") == "SessionBegins":
                    logger.info("Received SessionBegins event from Assembly AI")
                    yield create_ws_data_packet("speech_started", self.meta_info)

                elif msg.get("message_type") == "PartialTranscript":
                    transcript = msg.get("text", "")
                    if transcript.strip():
                        data = {
                            "type": "interim_transcript_received",
                            "content": transcript
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                elif msg.get("message_type") == "FinalTranscript":
                    transcript = msg.get("text", "")
                    if transcript.strip():
                        logger.info(f"Received final transcript from Assembly AI - {transcript}")
                        data = {
                            "type": "transcript",
                            "content": transcript
                        }
                        yield create_ws_data_packet(data, self.meta_info)

                elif msg.get("message_type") == "SessionTerminated":
                    logger.info(f"Received SessionTerminated from Assembly AI - {msg}")
                    self.meta_info["transcriber_duration"] = msg.get("duration", 0)
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

            except Exception as e:
                traceback.print_exc()
                self.interruption_signalled = False

    async def push_to_transcriber_queue(self, data_packet):
        """Push data packet to transcriber output queue"""
        await self.transcriber_output_queue.put(data_packet)

    async def assembly_connect(self):
        """Establish websocket connection to Assembly AI with proper error handling"""
        try:
            websocket_url = self.get_assembly_ws_url()
            additional_headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            logger.info(f"Attempting to connect to Assembly AI websocket: {websocket_url}")
            
            assembly_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, additional_headers=additional_headers),
                timeout=10.0  # 10 second timeout
            )
            
            self.websocket_connection = assembly_ws
            self.connection_authenticated = True
            logger.info("Successfully connected to Assembly AI websocket")
            
            return assembly_ws
            
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to Assembly AI websocket")
            raise ConnectionError("Timeout while connecting to Assembly AI websocket")
        except InvalidHandshake as e:
            logger.error(f"Invalid handshake during Assembly AI websocket connection: {e}")
            raise ConnectionError(f"Invalid handshake during Assembly AI websocket connection: {e}")
        except ConnectionClosedError as e:
            logger.error(f"Assembly AI websocket connection closed unexpectedly: {e}")
            raise ConnectionError(f"Assembly AI websocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Assembly AI websocket: {e}")
            raise ConnectionError(f"Unexpected error connecting to Assembly AI websocket: {e}")

    async def run(self):
        """Start the transcription task"""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Failed to start transcription task: {e}")

    def __calculate_utterance_end(self, data):
        """Calculate utterance end time"""
        utterance_end = None
        if 'words' in data:
            final_word = data['words'][-1]
            utterance_end = self.connection_start_time + final_word['end']
            logger.info(f"Final word ended at {utterance_end}")
        return utterance_end

    def __set_transcription_cursor(self, data):
        """Set transcription cursor position"""
        if 'words' in data:
            final_word = data['words'][-1]
            self.transcription_cursor = final_word['end']
        logger.info(f"Setting transcription cursor at {self.transcription_cursor}")
        return self.transcription_cursor

    def __calculate_latency(self):
        """Calculate latency between audio and transcription"""
        if self.transcription_cursor is not None:
            logger.info(f'audio cursor is at {self.audio_cursor} & transcription cursor is at {self.transcription_cursor}')
            return self.audio_cursor - self.transcription_cursor
        return None

    async def transcribe(self):
        """Main transcription method"""
        assembly_ws = None
        try:
            start_time = time.perf_counter()
            
            try:
                assembly_ws = await self.assembly_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish Assembly AI connection: {e}")
                await self.toggle_connection()
                return
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(assembly_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(assembly_ws))
                
                try:
                    async for message in self.receiver(assembly_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("closing the Assembly AI connection")
                            await self._close(assembly_ws, data={"type": "CloseStream"})
                            break
                except ConnectionClosedError as e:
                    logger.error(f"Assembly AI websocket connection closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise
            else:
                async for message in self.sender():
                    await self.push_to_transcriber_queue(message)

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if assembly_ws is not None:
                try:
                    await assembly_ws.close()
                    logger.info("Assembly AI websocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing websocket in finally block: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
            
            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
