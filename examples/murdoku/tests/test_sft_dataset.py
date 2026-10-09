import unittest

from murdoku_demo.sft_dataset import assistant_mask, tokenize_example


class SFTMaskTests(unittest.TestCase):
    def test_masks_prompt_and_tool_but_supervises_all_assistant_bodies(self):
        # 1/2 are message delimiters, 3/4 are assistant header tokens.
        ids = [1, 7, 8, 2, 1, 3, 4, 10, 11, 2, 1, 9, 12, 2, 1, 3, 4, 13, 2]
        mask, count = assistant_mask(ids, 1, 2, [3, 4])
        self.assertEqual(count, 2)
        self.assertEqual([t for t, m in zip(ids, mask) if m], [10, 11, 2, 13, 2])
        self.assertEqual(mask[10:14], [0, 0, 0, 0])

    def test_rejects_nested_chat_delimiters_in_tool_output(self):
        with self.assertRaises(ValueError):
            assistant_mask([1, 9, 1, 3, 4, 10, 2], 1, 2, [3, 4])

    def test_openai_arguments_are_parsed_without_mutating_trace(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
                assert kwargs["preserve_thinking"] is True
                return "recorded reasoning"

            def encode(self, text, **kwargs):
                return [3, 4] if text == "assistant\n" else [1, 3, 4, 10, 2]

            def convert_tokens_to_ids(self, text):
                return {"<|im_start|>": 1, "<|im_end|>": 2}[text]

        example = {
            "prompt_uid": "one",
            "tools": [],
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "recorded reasoning",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_python",
                                "arguments": '{"code":"print(1)"}',
                            }
                        }
                    ],
                }
            ],
        }
        tokenizer = Tokenizer()
        row = tokenize_example(example, tokenizer)
        self.assertEqual(tokenizer.arguments, {"code": "print(1)"})
        self.assertIsInstance(
            example["messages"][0]["tool_calls"][0]["function"]["arguments"], str
        )
        self.assertEqual(row["loss_mask"], [0, 0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
