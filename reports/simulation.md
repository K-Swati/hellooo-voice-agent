LLM: `mock` | today (simulated): 2026-11-16

| Scenario | Result | Caller turns | Tool calls | What it checks |
|---|---|---|---|---|
| `book_one_shot` | pass | 2 | 2 | Books in one message, then confirms |
| `book_incremental` | pass | 5 | 2 | Agent collects name, day and time one question at a time |
| `slot_taken` | pass | 3 | 2 | Requested slot is taken, caller accepts a nearby one |
| `weekend_request` | pass | 3 | 3 | Weekend is refused, caller picks a weekday |
| `reschedule` | pass | 3 | 3 | Moves an existing appointment |
| `cancel` | pass | 3 | 2 | Cancels an existing appointment |
| `decline_at_confirmation` | pass | 2 | 1 | Caller changes their mind at the read-back; nothing is booked |
| `unknown_patient` | pass | 1 | 1 | Cancel for a name with no appointment; nothing changes |
| `medical_advice` | pass | 1 | 1 | Medical advice request is handed to staff |
| `emergency` | pass | 1 | 1 | Emergency phrases skip the model and hand off immediately |
| `out_of_scope` | pass | 1 | 0 | Off-topic request is declined politely |

| Metric | Value |
|---|---|
| scenarios | 11 |
| task_success_rate | 1.0 |
| correct_final_state | 1.0 |
| correct_handoff | 1.0 |
| correct_changes | 1.0 |
| avg_caller_turns | 2.27 |
| avg_tool_calls | 1.64 |
| writes_blocked_by_guard | 0 |
