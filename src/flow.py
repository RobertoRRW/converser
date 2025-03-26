from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from agents.items import TResponseInputItem

from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.run import RunContextWrapper
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from agents import Agent

from converser.util import StructuredOutputVoiceWorkflow, record_audio, AudioPlayer
from agents.voice import AudioInput, VoicePipeline

load_dotenv()


class DialogueLine(BaseModel):
    dialogue: str



@dataclass
class ConversationState:
    conversation_language: str = "English"
    user_sentiment: Literal["Upset", "Neutral", "Pleased"] = "Neutral"
    initial_query: str | None = None
    issue_description: str | None = None
    urgency: str | None = None
    email: str | None = None
    device_type: str = "MISSING"
    model: str = "MISSING"
    brand: str = "MISSING"
    attempted_solutions: list[str] = field(default_factory=list)
    state_of_issue: str | None = None
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


greeter_agent = Agent(
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


def email_request_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
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


initial_agent = Agent(
    name="Email Request",
    model="gpt-4o-mini",
    instructions=email_request_instructions,
    output_type=InitialInfo | Unexpected | EndCall,  # type: ignore
)


class Email(BaseModel):
    """Email address provided by customer"""

    email: str = Field(description="Customer's registered email address")
    dialogue: str = Field(
        description="Agent dialogue confirming email and asking about device type"
    )


def email_validation_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.
        Your task is to validate the caller's email.
        Only validate that the email is correctly fromatted and plausible.
        If you could not validate the email, mark as unknown and
        ask for clarification in a natural way.
        
        Once you have the email address, ask the user for the following information:
            - Device type
            - Brand
            - Model
        ------
        Examples:

        Normal interaction:
        User: "john at company dot com"
        You: {{"email": "john@company.com", "dialogue": "Got it. Could tell me what sort of device you're having trouble with, brand and model?"}}

        Could not understand:
        User: "I like turtles"
        You: {{"type_": "unknown", "dialogue": "I'm going to need your email so we can continue"}}

        Invalid email:
        User: "It's john at company"
        You: {{"type_": "unknown", "dialogue": "Could you repeat that?"}}

        Something else:
        User: "Oh I forgot"
        You: {{"type_": "unknown", "dialogue": "We're going to need that in order to continue, maybe call back when you have it?"}}
        """


email_validation_agent = Agent(
    name="Email Request",
    model="gpt-4o-mini",
    instructions=email_validation_instructions,
    output_type=Email | Unexpected | EndCall,  # type: ignore
)


class DeviceInfo(BaseModel):
    """Information about customer's device"""

    device_type: str = Field(
        description="Type of device (laptop, desktop, smartphone, tablet, etc.)",
    )
    brand: str = Field(description="Brand/manufacturer of the device")
    model: str = Field(description="Model of device")
    user_confirmed: bool = Field(description="True if user confirmed the information")
    dialogue: str = Field(
        description="Agent dialogue asking for missing information or confirming complete info",
    )


def device_information_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.
        You are tasked with helping a user categorize their electronic devices for support purposes.
        The user will provide a text description of their device, and you must extract the Device Type, Brand, and Model.
        You must deduce the value of device_type from the user's description.
        The value you produce for device_type must exactly match (no extra characters) one of: 
            - smartphone
            - tablet
            - laptop
            - desktop
            - television
            - speaker
            - headphones
            - smartwatch
            - gaming_console
            - router
            - printer
            - camera
            - smart_home
            - unknown

        If any information is missing or cannot be determined from the text, fill the field with "MISSING".
        Incorrect or missing information could lead to the user's receiving misleading information, so accuracy is critical.

        If any information is MISSING, ask the user for clarification for that specific information.
        If the device_type is unknown, you may ask the caller for another description only once.

        If no values are MISSING, ask the user for confirmation.

        If the user confirms the information is correct, you must return the device information and ask the user to describe
        the problem in more detail.

        Information we have so far:
            - device_type: {context.context.device_type}
            - brand: {context.context.brand}
            - model: {context.context.model}
        """


device_agent = Agent(
    name="Device Information Agent",
    model="gpt-4o-mini",
    instructions=device_information_instructions,
    output_type=DeviceInfo | Unexpected | EndCall,  # type: ignore
)


class IssueDetails(BaseModel):
    """Detailed description of the technical issue"""

    issue_description: str = Field(
        description="Detailed description of the technical problem"
    )
    urgency: Literal["High", "Medium", "Low"] = Field(
        description="System-determined urgency level (not directly asked)"
    )
    dialogue: str = Field(
        description="Agent dialogue confirming issue details and offering next steps"
    )


def issue_information_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.
        Your job is to collect information about the user's problem.
        Your job IS NOT to debug or try to fix the issue.
        You only collect a description to pass it on to an expert.

        You must also determine the level of urgency yourself.
        Do NOT ask the caller the level of urgency directly.
        Urgency can be "High", "Medium", "Low"

        Finally ask if the user is fine waiting a short time to find a solution.
        ------
        Examples:

        Normal interaction:
        User: "Chrome gets very slow after using it for an hour"
        You: {{"issue_description": "Chrome is slow after an hour of use", "urgency": "Medium",
            "dialogue": "Can you stay on the line a couple of seconds while we find a solution?"}}

        Could not understand:
        User: "I like turtles"
        You: {{"type_": "unknown", "dialogue": "Could you clarify your specific tech problem?"}}

        Request more information:
        User: "My phone is slow"
        You: {{"type_": "uknnown", "dialogue": "Is this a general issue or does it happen only with certain apps?"}}
        """


issue_agent = Agent(
    model="gpt-4o-mini",
    name="Problem Definition Agent",
    instructions=issue_information_instructions,
    output_type=IssueDetails | Unexpected | EndCall,  # type: ignore
)


class SolutionResponse(BaseModel):
    """Response with potential solutions"""

    dialogue: str = Field(description="List of potential solutions to try")
    user_sentiment: Literal["Upset", "Neutral", "Pleased"] = Field(
        description="Current user sentiment"
    )


def solution_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.
        
        Your job is to give a plausible solution to the client's problem.
        You have the following information:
            - Issue description: {context.context.issue_description}
            - Device type: {context.context.device_type}
            - Brand: {context.context.brand}
            - Model: {context.context.model}

        Give the user troubleshooting steps.
        Do not give more than 4 short sentences.
        Take into account that the user's previous sentiment was {context.context.user_sentiment}.
        You must try to make the user feel pleased.
        
        You must also indicate whether the user's _current_ sentiment is: "Upset", "Neutral", "Pleased"
        Indicate the previous sentiment if there is no evidence that there has been a change of sentiment.

        Normal interaction:
        User: "Understood"
        You: {{"dialogue": [Your proposed solution here], user_sentiment: "{context.context.user_sentiment}"[No change]}}

        Could not understand:
        User: "This piece of crap is useless"
        You: {{"dialogue": [Your proposed solution here], user_sentiment: "Upset"}}
    """


solution_agent = Agent(
    model="gpt-4o",
    name="Solution Agent",
    instructions=solution_instructions,
    output_type=SolutionResponse | EndCall,  # type: ignore
)

class EscalationReply(BaseModel):
    """Check if customer is satisfied with solutions"""
    user_sentiment: Literal["Upset", "Neutral", "Pleased"] = Field(
        description="Current user sentiment"
    )
    state_of_issue: str = Field(description="Short description of the final state of the problem")
    dialogue: str = Field(description="Farewell dialogue")

def escalation_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.

        It has been decided that the caller's issue will be escalated. You must inform
        the caller that he will be at some other time. 

        You must indicate whether the user's _current_ sentiment is: "Upset", "Neutral", "Pleased"

        You must also give a short final description of what state the issue was left in.

        You will say farewell to the caller.
    """

escalation_agent = Agent(
    model="gpt-4o-mini",
    name="Escalation Agent",
    instructions=escalation_instructions,
    output_type=EscalationReply
)

def final_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.

        The user's issue has been resolved without escalation.

        You must also indicate whether the user's _current_ sentiment is: "Upset", "Neutral", "Pleased"

        You will now say goodbye.
    """

class FinalReply(BaseModel):
    """Check if customer is satisfied with solutions"""
    user_sentiment: Literal["Upset", "Neutral", "Pleased"] = Field(
        description="Current user sentiment"
    )
    dialogue: str = Field(description="Farewell dialogue")

final_agent = Agent(
    model="gpt-4o-mini",
    name="Final Agent",
    instructions=final_instructions,
    output_type=FinalReply,
)


def check_satisfaction_instructions(
    context: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return f"""
        {RECOMMENDED_PROMPT_PREFIX}
        You are a customer service agent for company TechService.
        Speak in {context.context.conversation_language}.
        
        Your job is to determine if the user is satisfied with the proposed solution.

        If the solution hasn't been satisfactory, you must consider what has been tried so far,
        and the user's sentiment to decide wether to escalate or to try another solution.
        
        If the user's problem is fixed, you will hand off to the sucess agent.
    """

satisfaction_triage = Agent(
    model="gpt-4o-mini",
    name="Satisfaction Triage",
    instructions=check_satisfaction_instructions,
    handoffs=[final_agent, solution_agent, escalation_agent],
)

async def audio_input(agent: Agent, ctx: GraphRunContext[ConversationState]) -> Any:
    workflow = StructuredOutputVoiceWorkflow[ConversationState](
        agent, "dialogue", ctx.state, ctx.state.message_history
    )
    pipeline = VoicePipeline(workflow=workflow)
    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    with AudioPlayer() as player:
        # Play the audio stream as it comes in
        async for event in result.stream():
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
        print("Greet")
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
        print("CollectEmail")
        result = await audio_input(initial_agent, ctx)
        match result:
            case InitialInfo(tech_issue=issue):
                ctx.state.initial_query = issue
                return ValidateEmail()
            case End():
                return Farewell()
            case _:
                return CollectEmail()


@dataclass
class ValidateEmail(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CollectDeviceInfo | ValidateEmail | Farewell:
        print("ValidateEmail")
        result = await audio_input(email_validation_agent, ctx)
        match result:
            case Email(email=email):
                ctx.state.email = email
                return CollectDeviceInfo()
            case End():
                return Farewell()
            case _:
                return ValidateEmail()


@dataclass
class CollectDeviceInfo(BaseNode[ConversationState]):
    rounds: int = 0

    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CollectIssueDetails | CollectDeviceInfo | Farewell:
        print("CollectDeviceInfo")
        if self.rounds >= 3:
            # Don't overwhelm the user, simply carry on
            return CollectIssueDetails()
        result = await audio_input(device_agent, ctx)

        match result:
            case DeviceInfo(user_confirmed=True):
                return CollectIssueDetails()
            case DeviceInfo(
                user_confirmed=False, device_type=device_type, brand=brand, model=model
            ):
                ctx.state.device_type = device_type
                ctx.state.brand = brand
                ctx.state.model = model
                return CollectDeviceInfo(self.rounds + 1)
            case End():
                return Farewell()
            case _:
                return CollectDeviceInfo(self.rounds + 1)


@dataclass
class CollectIssueDetails(BaseNode[ConversationState]):
    rounds: int = 0

    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> ProvideSolutions | CollectIssueDetails | Farewell:
        print("CollectIssueDetails")
        if self.rounds >= 3:
            # Don't overwhelm the user, simply carry on
            return ProvideSolutions()

        result = await audio_input(issue_agent, ctx)
        match result:
            case IssueDetails(issue_description=description, urgency=urgency):
                ctx.state.issue_description = description
                ctx.state.urgency = urgency
                return ProvideSolutions()
            case End():
                return Farewell()
            case _:
                return CollectIssueDetails(self.rounds + 1)


@dataclass
class ProvideSolutions(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> CheckSatisfaction | ProvideSolutions | Farewell:
        print("ProvideSolutions")
        # Placeholder for RAG - would query knowledge base here
        result = await audio_input(solution_agent, ctx)

        match result:
            case SolutionResponse(dialogue=solution, user_sentiment=sentiment):
                ctx.state.attempted_solutions += solution
                ctx.state.user_sentiment = sentiment
                return CheckSatisfaction()
            case End():
                return Farewell()
            case _:
                return ProvideSolutions()


@dataclass
class CheckSatisfaction(BaseNode[ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> Farewell | CheckSatisfaction:
        print("CheckSatisfaction")
        result = await audio_input(satisfaction_triage, ctx)
        match result:
            case FinalReply(user_sentiment=sentiment):
                ctx.state.user_sentiment=sentiment
                ctx.state.state_of_issue="Solved"
                return Farewell()
            case EscalationReply(user_sentiment=sentiment, state_of_issue=final_state):
                ctx.state.user_sentiment=sentiment
                ctx.state.state_of_issue=final_state
                return Farewell()
            case SolutionResponse(dialogue=solution, user_sentiment=sentiment):
                ctx.state.attempted_solutions += solution
                ctx.state.user_sentiment = sentiment
                return CheckSatisfaction()
            case End():
                return Farewell()
            case _:
                return CheckSatisfaction()



@dataclass
class Farewell(BaseNode[ConversationState, None, ConversationState]):
    async def run(
        self,
        ctx: GraphRunContext[ConversationState],
    ) -> End[ConversationState]:
        # Simple sink for the graph
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
    return result.output.message_history


if __name__ == "__main__":
    history = asyncio.run(main())
