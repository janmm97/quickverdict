"""Request schema, wire-compatible with TypeSafe's `POST /v1/systemone` (openapi 0.2.0)."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Content = str | dict[str, Any] | list[Any] | None


class _Q(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: Content = None


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    true: Content = None
    false: Content = None


class Noul(_Q):
    type: Literal["noul"]
    criteria: NoulCriteria | None = None


class Choice(_Q):
    type: Literal["choice"]
    criteria: Annotated[dict[str, Content], Field(min_length=1, max_length=255)]


class Score(_Q):
    type: Literal["score"]
    criteria: Annotated[list[str | dict[str, Any] | list[Any]], Field(min_length=1, max_length=10)]


class Box(_Q):
    type: Literal["box"]


Question = Annotated[Noul | Choice | Score | Box, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: str | dict[str, Any] | list[Any]
    questions: Annotated[dict[str, Question], Field(min_length=1)]
    assets: dict[str, Any] | None = None
