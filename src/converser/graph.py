from __future__ import annotations

from dataclasses import dataclass


from dotenv import load_dotenv
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from converser.agent import (
    ConversationState,
    DeviceInfo,
    Email,
    EscalationReply,
    FinalReply,
    Greeting,
    InitialInfo,
    IssueDetails,
    SolutionResponse,
    device_agent,
    greeter_agent,
    issue_agent,
    solution_agent,
    satisfaction_triage,
    initial_agent,
    email_validation_agent,
)
from converser.audio import audio_input

load_dotenv()


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
                ctx.state.user_sentiment = sentiment
                ctx.state.state_of_issue = "Solved"
                return Farewell()
            case EscalationReply(user_sentiment=sentiment, state_of_issue=final_state):
                ctx.state.user_sentiment = sentiment
                ctx.state.state_of_issue = final_state
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

def _make_graph():
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
    return cs_graph

def make_mermaid() -> str:
    cs_graph = _make_graph()
    return cs_graph.mermaid_code()

async def run():
    cs_graph = _make_graph()
    state = ConversationState()
    result = await cs_graph.run(Greet(), state=state)
    return result.output
