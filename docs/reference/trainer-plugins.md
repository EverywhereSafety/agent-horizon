# Trainer extensions

The public core provides trajectory accounting and built-in GRPO/OPD adapters.
Custom objectives live in separately installed packages. The core does not import
or distribute their implementations, qualification scripts or algorithm documents.

Register a Python package entry point in `long_horizon_rl.trainers` whose name
matches `trainer.v1.trainer_mode`. Its value is a zero-argument installer that
registers the trainer with the pinned veRL runtime. Only the selected installer
is loaded, inside the Ray task runner before upstream trainer construction.

```toml
[project.entry-points."long_horizon_rl.trainers"]
long_horizon_custom = "custom_package.plugin:install"
```

An installer for a separate-async trainer must set
`install.requires_async_checkpoint = True`. This preserves the core's prefetch
validation and TransferQueue checkpoint requirement for the extension.
Missing `long_horizon_*` plugins and ambiguous registrations fail explicitly;
other upstream trainer modes remain available through veRL.

A plugin owns its loss, algorithm-specific settings, checkpoint identity and
qualification tests. Environment plugins remain a separate interface: choosing
an algorithm must not require importing a particular task environment.
