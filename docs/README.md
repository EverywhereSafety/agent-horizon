# Agent Horizon documentation

[Website](https://everywheresafety.github.io/agent-horizon/) ·
[Blog](https://everywheresafety.github.io/blog/murdoku-as-vhd/) ·
[Installation and training quickstart](../README.md)

Start with [how episodes continue across context windows](../README.md#how-it-works).
Then connect your task through the [environment interface](guides/environments.md)
and follow the [training quickstart](../README.md#get-started). Runtime setup and
qualification commands live in [runtime checks](reference/runtime-checks.md).

## Guides

| I want to… | Guide |
| --- | --- |
| Supply queries, dynamics, tools and rewards | [Environment interface](guides/environments.md) |
| Configure context clearing, memory and capacity | [Context](guides/context.md) |
| Run asynchronous rollout | [Async execution](guides/async.md) |
| Resume episodes and handle cancellation | [Recovery](guides/recovery.md) |
| Use on-policy distillation | [OPD](guides/opd.md) |

## Reference

- [Architecture and module map](reference/architecture.md)
- [Runtime checks](reference/runtime-checks.md)
- [Trainer extensions](reference/trainer-plugins.md)
