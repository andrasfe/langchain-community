"""Unit tests for DALL-E Image Generator utility."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper


class FakeOpenAIResponseData:
    """A fake response data object."""

    def __init__(self, *, url: str):
        self.url = url


class FakeOpenAIResponse:
    """A fake response object."""

    def __init__(self, *, data: list[FakeOpenAIResponseData]):
        self.data = data


class FakeImages:
    """A fake images client."""

    def generate(self, *args: Any, **kwargs: Any) -> FakeOpenAIResponse:
        """A fake generate method."""
        # This can be expanded to check kwargs if needed
        if "n" in kwargs and kwargs["n"] > 1:
            urls = [f"https://example.com/image_{i}.png" for i in range(kwargs["n"])]
            return FakeOpenAIResponse(
                data=[FakeOpenAIResponseData(url=u) for u in urls]
            )
        return FakeOpenAIResponse(
            data=[FakeOpenAIResponseData(url="https://example.com/image.png")]
        )


class FakeOpenAI:
    """A fake OpenAI client."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.images = FakeImages()


@pytest.fixture(autouse=True)
def mock_openai_v1():
    """Mock the OpenAI v1 check and OpenAI client."""
    with (
        patch("openai.OpenAI", FakeOpenAI),
        patch(
            "langchain_community.utilities.dalle_image_generator.is_openai_v1",
            return_value=True,
        ) as mock_is_v1,
    ):
        yield mock_is_v1


def test_dalle2_excludes_quality():
    """Test that quality parameter is excluded for dall-e-2."""
    with patch.object(
        FakeImages, "generate", wraps=FakeImages().generate
    ) as mock_generate:
        prompt = "a renaissance style photo of a cat"
        wrapper = DallEAPIWrapper(
            model="dall-e-2", quality="hd", api_key=SecretStr("foo")
        )
        wrapper.run(prompt)
        mock_generate.assert_called_once()
        called_kwargs = mock_generate.call_args.kwargs
        assert "quality" not in called_kwargs
        assert called_kwargs["model"] == "dall-e-2"


def test_dalle3_includes_quality():
    """Test that quality parameter is included for dall-e-3."""
    with patch.object(
        FakeImages, "generate", wraps=FakeImages().generate
    ) as mock_generate:
        prompt = "a pixel art style photo of a dog"
        wrapper = DallEAPIWrapper(
            model="dall-e-3", quality="hd", api_key=SecretStr("foo")
        )
        wrapper.run(prompt)
        mock_generate.assert_called_once()
        called_kwargs = mock_generate.call_args.kwargs
        assert called_kwargs["quality"] == "hd"
        assert called_kwargs["model"] == "dall-e-3"


def test_multiple_image_urls_response():
    """Test that multiple image URLs are correctly returned with a separator."""
    prompt = "two cats playing"
    wrapper = DallEAPIWrapper(n=2, separator="|", api_key=SecretStr("foo"))
    result = wrapper.run(prompt)
    assert result == "https://example.com/image_0.png|https://example.com/image_1.png"


def test_single_image_url_response():
    """Test that a single image URL is correctly returned."""
    prompt = "a single beautiful flower"
    wrapper = DallEAPIWrapper(api_key=SecretStr("foo"))
    result = wrapper.run(prompt)
    assert result == "https://example.com/image.png"


def test_no_image_generated_response():
    """Test the response when no image is generated."""
    with patch.object(FakeImages, "generate") as mock_generate:
        mock_generate.return_value = FakeOpenAIResponse(data=[])
        prompt = "an impossible image"
        wrapper = DallEAPIWrapper(api_key=SecretStr("foo"))
        result = wrapper.run(prompt)
        assert result == "No image was generated"
        mock_generate.assert_called_once()


def test_extra_moved_to_model_kwargs():
    """Test that extra parameters are moved to model_kwargs."""
    wrapper = DallEAPIWrapper(api_key=SecretStr("foo"), invalid_parameter="value")  # type: ignore[call-arg]
    assert wrapper.model_kwargs == {"invalid_parameter": "value"}


@pytest.mark.parametrize(
    "model,quality,should_include_quality",
    [
        ("dall-e-2", "hd", False),
        ("dall-e-2", "standard", False),
        ("dall-e-3", "hd", True),
        ("dall-e-3", "standard", True),
    ],
)
def test_quality_parameter_handling(model, quality, should_include_quality):
    """Test that quality parameter is handled correctly for different models."""
    with patch.object(
        FakeImages, "generate", wraps=FakeImages().generate
    ) as mock_generate:
        prompt = "test image"
        wrapper = DallEAPIWrapper(
            model=model, quality=quality, api_key=SecretStr("foo")
        )
        wrapper.run(prompt)
        mock_generate.assert_called_once()
        called_kwargs = mock_generate.call_args.kwargs

        if should_include_quality:
            assert called_kwargs["quality"] == quality
        else:
            assert "quality" not in called_kwargs

        assert called_kwargs["model"] == model
