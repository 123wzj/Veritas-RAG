# -*- coding: utf-8 -*-
"""
联网搜索服务
支持多种搜索提供商
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import httpx
try:
    from core.config import settings
except ImportError:  # pragma: no cover - fallback for package-style imports
    from backend.core.config import settings


class BaseSearchProvider(ABC):
    """搜索提供商基类"""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        执行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        pass


class TavilySearchProvider(BaseSearchProvider):
    """Tavily 搜索提供商"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """使用 Tavily 进行搜索"""
        from tavily import TavilyClient

        try:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                    "score": item.get("score", 0.8),
                })

            return results

        except Exception as e:
            print(f"Tavily search error: {e}")
            return []


class SerpApiSearchProvider(BaseSearchProvider):
    """SerpApi 搜索提供商"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.webscrapeapi.com/search"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """使用 SerpApi 进行搜索"""
        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "num": max_results,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                    "score": 0.8,
                })

            return results

        except Exception as e:
            print(f"SerpApi search error: {e}")
            return []


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """DuckDuckGo 搜索提供商（免费，无需 API Key）"""

    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """使用 DuckDuckGo 进行搜索"""
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = []

            async with httpx.AsyncClient() as client:
                # DuckDuckGo 不需要 API key
                search_url = f"https://html.duckduckgo.com/html/?q={query}"
                response = await client.get(search_url, timeout=30)

            # 简化实现：返回空结果
            # TODO: 实现真正的 DuckDuckGo 解析
            return []

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []


class WebSearchService:
    """联网搜索服务"""

    def __init__(self):
        self._provider: Optional[BaseSearchProvider] = None

    def get_provider(self) -> BaseSearchProvider:
        """获取配置的搜索提供商"""
        if self._provider:
            return self._provider

        provider_name = settings.WEB_SEARCH_PROVIDER.lower()

        if provider_name == "tavily":
            self._provider = TavilySearchProvider(settings.WEB_SEARCH_API_KEY)
        elif provider_name == "serpapi":
            self._provider = SerpApiSearchProvider(settings.WEB_SEARCH_API_KEY)
        elif provider_name == "duckduckgo":
            self._provider = DuckDuckGoSearchProvider()
        else:
            # 默认使用 DuckDuckGo（免费）
            self._provider = DuckDuckGoSearchProvider()

        return self._provider

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        执行网络搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        provider = self.get_provider()
        return await provider.search(query, max_results)

    async def search_with_snippets(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        执行搜索并获取完整摘要

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            包含完整摘要的搜索结果列表
        """
        results = await self.search(query, max_results)

        # 如果需要，可以进一步获取网页内容
        # TODO: 实现网页抓取和内容提取

        return results


# 全局搜索服务实例
web_search_service = WebSearchService()
