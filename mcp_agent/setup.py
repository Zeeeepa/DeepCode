"""
Setup script for MCP Agent framework.
"""

from setuptools import setup, find_packages

setup(
    name="mcp_agent",
    version="0.1.13",
    description="Model Context Protocol Agent Framework",
    author="DeepCode",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "asyncio",
    ],
    python_requires=">=3.8",
)

