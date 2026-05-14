"""Fixture for call-graph extraction tests.

Known call graph:
    main          -> helper, build_widget, Widget.render
    helper        -> validate
    Widget.render -> self.format, format_html
    Widget.format -> escape

    escape, format_html, validate — leaf functions (no calls).
"""
from __future__ import annotations


def escape(s: str) -> str:
    return s


def format_html(html: str) -> str:
    return escape(html)


def validate(x: int) -> bool:
    return x > 0


class Widget:
    def render(self, content: str) -> str:
        body = self.format(content)
        return format_html(body)

    def format(self, content: str) -> str:
        return escape(content)


def build_widget() -> Widget:
    return Widget()


def helper(n: int) -> bool:
    return validate(n)


def main() -> str:
    if helper(42):
        w = build_widget()
        return w.render("hello")
    return ""
