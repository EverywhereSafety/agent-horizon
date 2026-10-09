import tempfile
import unittest
from pathlib import Path
from murdoku_demo.visual_sft import resolve_messages


class VisualRecordTests(unittest.TestCase):
    def test_image_resolution_preserves_source_and_excludes_hidden_fields(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "board.png").write_bytes(b"fixture")
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "board.png"},
                            {"type": "text", "text": "Solve"},
                        ],
                    }
                ],
                "hidden_answer": "SECRET",
            }
            messages, images = resolve_messages(example, root)
            self.assertEqual(example["messages"][0]["content"][0]["image"], "board.png")
            self.assertTrue(images[0].is_absolute())
            self.assertNotIn("SECRET", str(messages))

    def test_missing_image_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ["missing.png", "../outside.png", "/tmp/image.png"]:
                example = {
                    "messages": [
                        {"role": "user", "content": [{"type": "image", "image": path}]}
                    ]
                }
                with self.subTest(path=path), self.assertRaises(ValueError):
                    resolve_messages(example, root)

    def test_tool_arguments_are_normalized(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.png").touch()
            example = {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": "a.png"}]},
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": '{"code":"x"}'}}],
                    },
                ]
            }
            messages, _ = resolve_messages(example, root)
            self.assertEqual(
                messages[1]["tool_calls"][0]["function"]["arguments"], {"code": "x"}
            )

    def test_text_only_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            resolve_messages({"messages": [{"role": "user", "content": "text"}]}, ".")


class ScreenshotFormatTests(unittest.TestCase):
    def test_openai_relative_image_url(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "a.png").touch()
            messages, paths = resolve_messages(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "a.png"}}
                            ],
                        }
                    ]
                },
                root,
            )
            self.assertEqual(messages[0]["content"][0]["type"], "image")
            self.assertEqual(paths[0].name, "a.png")

    def test_tool_screenshot_data_url(self):
        import base64

        data = base64.b64encode(b"png-fixture").decode()
        messages, images = resolve_messages(
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + data},
                            }
                        ],
                    }
                ]
            },
            ".",
        )
        self.assertEqual(images[0].getvalue(), b"png-fixture")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
