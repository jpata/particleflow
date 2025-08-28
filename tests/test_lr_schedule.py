import torch
from torch.optim import SGD
from mlpf.model.utils import get_lr_schedule, save_checkpoint, load_checkpoint, load_lr_schedule
import os

def test_lr_schedule_restoration():
    # 1. Dummy model and optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=0.1)

    # 2. Original scheduler
    total_steps = 100
    config = {"lr_schedule": "cosinedecay", "lr": 0.1}
    original_scheduler = get_lr_schedule(config, optimizer, total_steps)

    # 3. Simulate training (Part 1)
    original_lrs = []
    for _ in range(50):
        original_lrs.append(original_scheduler.get_last_lr()[0])
        original_scheduler.step()

    # 4. Save checkpoint
    extra_state = {"lr_schedule_state_dict": original_scheduler.state_dict()}
    checkpoint_path = "test_checkpoint.pth"
    save_checkpoint(checkpoint_path, model, optimizer, extra_state)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # 5. New scheduler (Simulating Bug)
    # The scheduler is initialized with the *original* total_steps, but the last_epoch is set to the checkpoint step
    incorrect_scheduler = get_lr_schedule(config, optimizer, total_steps, last_batch=50)

    # 6. Load checkpoint (this step is actually redundant if we initialize with last_batch, but keeping it for analogy)
    # The load_lr_schedule function also sets last_epoch, so this is effectively what happens.
    load_lr_schedule(incorrect_scheduler, checkpoint, 50)

    # 7. Simulate training (Part 2)
    incorrect_lrs = []
    for _ in range(50):
        incorrect_lrs.append(incorrect_scheduler.get_last_lr()[0])
        incorrect_scheduler.step()

    # Continue original scheduler to get expected LRs
    for _ in range(50):
        original_lrs.append(original_scheduler.get_last_lr()[0])
        original_scheduler.step()

    print("original_lrs", original_lrs[50:])

    # 8. Assert failure
    print("incorrect_lrs", incorrect_lrs)
    assert original_lrs[50:] != incorrect_lrs

    # 9. New scheduler (Correct)
    correct_scheduler = get_lr_schedule(config, optimizer, total_steps)

    # 10. Load checkpoint
    load_lr_schedule(correct_scheduler, checkpoint, 50)

    # 11. Simulate training (Correct)
    correct_lrs = []
    for _ in range(50):
        correct_lrs.append(correct_scheduler.get_last_lr()[0])
        correct_scheduler.step()

    print("correct_lrs", correct_lrs)
    assert original_lrs[50:] == correct_lrs

    os.remove(checkpoint_path)