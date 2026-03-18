# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `mlpf/standalone/eval.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `mlpf/standalone/train.py` — you can modify this. Model architecture setup. Try to modify the architecture in creative ways, for example experiment with the attention backbone structure, or different linear or approximate attention strategies.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed total time budget of 60 seconds** (comprised of 3 training runs of 20 seconds each, with dataset shuffling). You launch it simply as: `./scripts/local/train.sh`

**What you CAN do:**
- Modify `mlpf/standalone/train.py` — this is the only file you edit. Everything is fair game: model architecture, hyperparameters size, model size, etc.

**What you CANNOT do:**
- Modify `mlpf/standalone/eval.py`. It is read-only.
- Install new packages or add dependencies. 
- Modify the evaluation code.

**The goal is simple: get the lowest validation jet interquartile range and lowest model runtime on CPU and GPU.** Since the time budget is fixed, you don't need to worry about training time — it's always 60 seconds (3x20s). Everything is fair game: change the architecture, the hyperparameters, the batch size, the model size, the optimizer, the loss configuration. In particular, focus on creative architectural exploration beyond just changing the hyperparameter values. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful validation jet iqr gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 validation jet iqr improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 validation jet iqr improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary of the 3 runs like this:

```
--- Final Results (3 runs) ---
val_jet_iqr     : 0.997900 ± 0.000001 (var)
training_seconds: 20.000000 ± 0.000000 (var)
total_seconds   : 25.000000 ± 0.000001 (var)
peak_vram_mb    : 100.000000 ± 0.000000 (var)
num_steps       : 100.000000 ± 0.000000 (var)
runtime_cpu_ms  : 10.000000 ± 0.000000 (var)
runtime_gpu_ms  : 5.000000 ± 0.000000 (var)

num_params_M:     0.1
depth:            3
```

Note that the script is configured to always stop after a fixed duration per run, so depending on the computing platform of this computer the numbers might look different. You can extract the mean metrics from the log file:

```
grep "^val_jet_iqr" run.log
```

## Logging results

When an experiment is done, log the **mean values** from the final results to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_jet_iqr	runtime_cpu_ms	runtime_gpu_ms	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. mean val_jet_iqr achieved (e.g. 1.234567) — use 0.000000 for crashes
3. mean runtime_cpu_ms in ms
4. mean runtime_gpu_ms in ms
5. mean peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	val_jet_iqr	runtime_cpu_ms	runtime_gpu_ms	memory_gb	status	description
a1b2c3d	0.997900	138	11	44.0	keep	baseline
b2c3d4e	0.993200	242	12	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	433	13	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	152	15	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `mlpf/model/mlpf.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: ./scripts/local/train.sh > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_jet_iqr:\|^peak_vram_mb:\|^runtime_cpu_ms:\|^runtime_gpu_ms:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_jet_iqr improved (lower), you "advance" the branch, keeping the git commit
9. If val_jet_iqr is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 10/hour, for a total of about 80 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

