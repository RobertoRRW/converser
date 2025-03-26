import curses
import time

from agents.voice.input import AudioInput
from agents.voice.pipeline import VoicePipeline
import numpy as np
import numpy.typing as npt
from pydantic_graph import GraphRunContext
import sounddevice as sd

from agents import Agent
from typing import Any, AsyncIterator
from agents.voice.workflow import VoiceWorkflowBase, VoiceWorkflowHelper
from agents.items import TResponseInputItem
from agents import Runner
from converser.agent import ConversationState
from converser.parser import filter_json_field


class StructuredOutputVoiceWorkflow[T](VoiceWorkflowBase):
    """A simple voice workflow that runs a single agent. Each transcription and result is added to
    the input history.
    For more complex workflows (e.g. multiple Runner calls, custom message history, custom logic,
    custom configs), subclass `VoiceWorkflowBase` and implement your own logic.
    """

    def __init__(
        self,
        agent: Agent[T],
        dialogue_field: str,
        context: T | None = None,
        history: list[TResponseInputItem] | None = None,
    ):
        """Create a new single agent voice workflow.

        Args:
            agent: The agent to run.
            callbacks: Optional callbacks to call during the workflow.
        """
        self.history: list[TResponseInputItem] = [] if history is None else history
        self._current_agent = agent
        self.dialogue_field = dialogue_field
        self.context = context

    async def run(self, transcription: str) -> AsyncIterator[str]:
        # Add the transcription to the input history
        self.history.append(
            {
                "role": "user",
                "content": transcription,
            }
        )
        print(f"heard: {transcription}")

        # Run the agent
        result = Runner.run_streamed(
            self._current_agent, self.history, context=self.context
        )

        # Stream the text from the result
        filtered_tokens = filter_json_field(
            VoiceWorkflowHelper.stream_text_from(result), self.dialogue_field
        )
        async for token in filtered_tokens:
            yield token

        # Update the input history and current agent
        self.history = result.to_input_list()
        self._current_agent = result.last_agent
        self.result = result


async def audio_input(agent: Agent, ctx: GraphRunContext[ConversationState]) -> Any:
    workflow = StructuredOutputVoiceWorkflow[ConversationState](
        agent, "dialogue", ctx.state, ctx.state.message_history
    )
    pipeline = VoicePipeline(workflow=workflow)
    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    with AudioPlayer() as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.add_audio(event.data)  # type: ignore

    ctx.state.message_history = workflow.history

    result = workflow.result.final_output
    return result


# Code below this line is Copyright (c) 2025 OpenAI
def _record_audio(screen: curses.window) -> npt.NDArray[np.float32]:
    screen.nodelay(True)  # Non-blocking input
    screen.clear()
    screen.addstr(
        "Press <spacebar> to start recording. Press <spacebar> again to stop recording.\n"
    )
    screen.refresh()

    recording = False
    audio_buffer: list[npt.NDArray[np.float32]] = []

    def _audio_callback(indata, frames, time_info, status):
        if status:
            screen.addstr(f"Status: {status}\n")
            screen.refresh()
        if recording:
            audio_buffer.append(indata.copy())

    # Open the audio stream with the callback.
    with sd.InputStream(
        samplerate=24000, channels=1, dtype=np.float32, callback=_audio_callback
    ):
        while True:
            key = screen.getch()
            if key == ord(" "):
                recording = not recording
                if recording:
                    screen.addstr("Recording started...\n")
                else:
                    screen.addstr("Recording stopped.\n")
                    break
                screen.refresh()
            time.sleep(0.01)

    # Combine recorded audio chunks.
    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
    else:
        audio_data = np.empty((0,), dtype=np.float32)

    return audio_data


def record_audio():
    # Using curses to record audio in a way that:
    # - doesn't require accessibility permissions on macos
    # - doesn't block the terminal
    audio_data = curses.wrapper(_record_audio)
    return audio_data


class AudioPlayer:
    def __enter__(self):
        self.stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.stop()  # wait for the stream to finish
        self.stream.close()

    def add_audio(self, audio_data: npt.NDArray[np.int16]):
        self.stream.write(audio_data)
