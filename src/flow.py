from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from agents.items import TResponseInputItem

from agents.run import RunContextWrapper
from agents.voice.workflow import VoiceWorkflowBase
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from agents import Agent as OAI_Agent

from converser.util import StructuredOutputVoiceWorkflow, record_audio, AudioPlayer
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

load_dotenv()


class DialogueLine(BaseModel):
    dialogue: str


parrot_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a customer service agent for company TechService, say a variation of the following using a different expression: ",
    result_type=DialogueLine,
)


@dataclass
class ConversationState:
    conversation_language: str = "English"
    initial_query: str | None = None
    message_history: list[TResponseInputItem] = field(default_factory=list)


class Unexpected(BaseModel):
    """Return if user's response is unclear"""

    type_: Literal["unexpected"]
    dialogue: str = Field(description="Dialogue requesting clarification")


class EndCall(BaseModel):
    """Return this if user expresses desire to end the call"""

    type_: Literal["end_request"]
    dialogue: str = Field(description="Farewell dialogue")


class Greeting(BaseModel):
    detected_language: str
    dialogue: str


greeter_agent = OAI_Agent(
    name="Greeter",
    model="gpt-4o-mini",
    handoff_description="Specialist agent for math questions",
    instructions="You are a customer service agent for company TechService, say an appropriate customer service greeting. "
    "Use the same language as the caller.",
    output_type=Greeting,
)


class InitialInfo(BaseModel):
    tech_issue: str = Field(description="Short summary of the caller's tech issue")
    dialogue: str = Field(description="Agent dialogue requesting user's email")


initial_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a customer service agent for company TechService. "
    "Register relevant information for tech support. "
    "Ask for user's registered email address to continue",
    result_type=InitialInfo | Unexpected | EndCall,  # type: ignore
)


def email_requester_instructions(
    context: RunContextWrapper[ConversationState], agent: OAI_Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.
        Your task is to extract an initial description of the caller's problem.
        It is not your job to debug the issue. Only an initial description.
        After you get this, ask for the user's registered email.
        If you could not collect the required information, mark as unknown and
        ask for clarification
        ------
        Examples:

        Normal interaction:
        User: "My phone is slow"
        You: {{"tech_issue": "slow phone", "dialogue": "Okay, coul you tell me the email address you have registered with us"}}

        Could not understand:
        User: "I like turtles"
        You: {{"type_": "unknown", "dialogue": "Apologies but this is a tech support line, do you have a tech support problem?"}}
        """


initial_agent_oai = OAI_Agent(
    name="Email Request",
    model="gpt-4o-mini",
    instructions=email_requester_instructions,
    output_type=InitialInfo | Unexpected | EndCall,  # type: ignore
)


class Email(BaseModel):
    """Email address provided by customer"""

    email: str = Field(description="Customer's registered email address")
    agent_response: str = Field(
        description="Agent dialogue confirming email and asking about device type"
    )


email_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a customer service agent for TechService. "
    "Identify if the user has provided an email address. "
    "If so, extract and validate it (must contain @ symbol). "
    "Then ask about their device type, brand and model.",
    result_type=Email | Unexpected | EndCall,  # type: ignore
)


class DeviceInfo(BaseModel):
    """Information about customer's device"""

    device_type: str | None = Field(
        default=None,
        description="Type of device (laptop, desktop, smartphone, tablet, etc.)",
    )
    brand: str | None = Field(
        default=None, description="Brand/manufacturer of the device"
    )
    model: str | None = Field(default=None, description="Model of device")
    agent_response: str = Field(
        default="",
        description="Agent dialogue asking for missing information or confirming complete info",
    )

    def missing_data(self) -> bool:
        return not all((self.device_type, self.brand, self.model))

    def show_state(self) -> str:
        return f"""
            - Device type: {self.device_type or "Not provided"}
            - Brand: {self.brand or "Not provided"}
            - Model: {self.model or "Not provided"}
        """


device_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="""
        You are a customer service agent for TechService.
        Extract information about the customer's device including device type, brand, and model.
        Check which fields have been provided and which are still missing.
        If all information is complete, confirm understanding and ask about their technical issue.
        If any device information is missing, specifically follow up asking for the missing pieces.
    """,
    result_type=DeviceInfo | Unexpected | EndCall,  # type: ignore
)


class IssueDetails(BaseModel):
    """Detailed description of the technical issue"""

    issue_description: str = Field(
        description="Detailed description of the technical problem"
    )
    urgency: str = Field(
        description="System-determined urgency level (not directly asked)"
    )
    agent_response: str = Field(
        description="Agent dialogue confirming issue details and offering next steps"
    )


issue_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a customer service agent for TechService. "
    "Extract detailed information about the customer's technical issue. "
    "Determine urgency based on issue description (do NOT ask customer directly). "
    "Confirm understanding and prepare to offer solutions.",
    result_type=IssueDetails | Unexpected | EndCall,  # type: ignore
)


class SolutionResponse(BaseModel):
    """Response with potential solutions"""

    solutions: list[str] = Field(description="List of potential solutions to try")
    agent_response: str = Field(description="Agent dialogue explaining solutions")


solution_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a customer service agent for TechService. "
    "Based on the device information and issue description, provide potential solutions. "
    #  "For now, generate plausible solutions without RAG. "
    "Offer step-by-step instructions for the customer to try.",
    result_type=SolutionResponse | Unexpected | EndCall,  # type: ignore
)


class SatisfactionCheck(BaseModel):
    """Check if customer is satisfied with solutions"""

    is_satisfied: bool = Field(
        description="Whether customer indicates satisfaction with solution"
    )
    agent_response: str = Field(description="Agent confirmation or escalation message")


satisfaction_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a customer service agent for TechService. "
    "Determine if the customer is satisfied with the solutions provided. "
    "If satisfied, prepare to close the conversation. "
    "If not satisfied, prepare to offer escalation options.",
    result_type=SatisfactionCheck | Unexpected | EndCall,  # type: ignore
)


async def audio_input(agent: OAI_Agent, ctx: GraphRunContext[ConversationState]) -> Any:
    workflow = StructuredOutputVoiceWorkflow[ConversationState](
        agent, "dialogue", ctx.state
    )
    pipeline = VoicePipeline(workflow=workflow)
    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    with AudioPlayer() as player:
        # Play the audio stream as it comes in
        async for event in result.stream():
            print(event)
            if event.type == "voice_stream_event_audio":
                player.add_audio(event.data)  # type: ignore

    ctx.state.message_history = workflow.history

    result = workflow.result.final_output
    return result


@dataclass
class Greet(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CollectEmail | Greet | Farewell:
        result = await audio_input(greeter_agent, ctx)
        match result:
            case Greeting(detected_language=language):
                ctx.state.conversation_language = language
                return CollectEmail()
            case End():
                return Farewell()
            case _:
                return Greet()


@dataclass
class CollectEmail(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> ValidateEmail | CollectEmail | Farewell:
        result = await audio_input(initial_agent_oai, ctx)
        match result:
            case InitialInfo(tech_issue=issue):
                ctx.state.initial_query = issue
                return ValidateEmail()
            case End():
                return CollectEmail()
            case _:
                return Farewell()


@dataclass
class ValidateEmail(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CollectDeviceInfo | ValidateEmail | Farewell:
        query = input()
        result = await email_agent.run(
            f"{query}"  # , message_history=ctx.state.message_history
        )
        #        ctx.state.message_history += result.all_messages()

        if isinstance(result.data, Email):
            print(result.data.agent_response)
            return CollectDeviceInfo(DeviceInfo())
        elif isinstance(result.data, Unexpected):
            print(result.data.clarification_request)
            return ValidateEmail()
        else:  # EndCall
            print("Call end requested")
            return Farewell()


@dataclass
class CollectDeviceInfo(BaseNode[ConversationState]):
    device_info: DeviceInfo

    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CollectIssueDetails | CollectDeviceInfo | Farewell:
        query = input()
        prompt = f"""
            Current information:
            {self.device_info.show_state()}

            User message: {query}
        """
        print(prompt)
        result = await device_agent.run(
            f"{prompt}", message_history=ctx.state.message_history
        )
        ctx.state.message_history += result.all_messages()

        if isinstance(result.data, DeviceInfo):
            print(result.data.agent_response)
            if result.data.missing_data():
                return CollectDeviceInfo(result.data)
            print(result.data.show_state())
            return CollectIssueDetails()
        elif isinstance(result.data, Unexpected):
            print(result.data.clarification_request)
            return CollectDeviceInfo(self.device_info)
        else:  # EndCall
            print("Call end requested")
            return Farewell()


@dataclass
class CollectIssueDetails(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> ProvideSolutions | CollectIssueDetails | Farewell:
        query = input()
        result = await issue_agent.run(
            f"{query}", message_history=ctx.state.message_history
        )
        ctx.state.message_history += result.all_messages()

        if isinstance(result.data, IssueDetails):
            print(result.data.agent_response)
            return ProvideSolutions()
        elif isinstance(result.data, Unexpected):
            print(result.data.clarification_request)
            return CollectIssueDetails()
        else:  # EndCall
            print("Call end requested")
            return Farewell()


@dataclass
class ProvideSolutions(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CheckSatisfaction | ProvideSolutions | Farewell:
        # Placeholder for RAG - would query knowledge base here
        # For now, we'll use the solution agent to generate plausible solutions
        result = await solution_agent.run(
            "Based on the information provided, what solutions can you suggest?",
            message_history=ctx.state.message_history,
        )
        ctx.state.message_history += result.all_messages()

        if isinstance(result.data, SolutionResponse):
            print(result.data.agent_response)
            print(result.data.solutions)
            return CheckSatisfaction()
        elif isinstance(result.data, Unexpected):
            print(result.data.clarification_request)
            return ProvideSolutions()
        else:  # EndCall
            print("Call end requested")
            return Farewell()


@dataclass
class CheckSatisfaction(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> Farewell | CheckSatisfaction | ProvideSolutions:
        query = input()
        result = await satisfaction_agent.run(
            f"{query}", message_history=ctx.state.message_history
        )
        ctx.state.message_history += result.all_messages()

        if isinstance(result.data, SatisfactionCheck):
            print(result.data.agent_response)
            if result.data.is_satisfied:
                return Farewell()
            else:
                # Here you could add an escalation node in a more complete system
                # For now, we'll try providing more solutions
                print("Let me see if I can suggest some additional solutions.")
                return ProvideSolutions()
        elif isinstance(result.data, Unexpected):
            print(result.data.clarification_request)
            return CheckSatisfaction()
        else:  # EndCall
            print("Call end requested")
            return Farewell()


@dataclass
class Farewell(BaseNode[ConversationState, None, ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> End[ConversationState]:
        result = await parrot_agent.run("Have a great day")
        ctx.state.message_history += result.all_messages()
        print(result.data.dialogue)
        return End(ctx.state)


async def main():
    state = ConversationState()
    cs_graph = Graph(
        nodes=(
            Greet,
            CollectEmail,
            ValidateEmail,
            CollectDeviceInfo,
            CollectIssueDetails,
            ProvideSolutions,
            CheckSatisfaction,
            Farewell,
        )
    )
    result = await cs_graph.run(Greet(), state=state)
    print(result.output.initial_query)


if __name__ == "__main__":
    asyncio.run(main())
