# Skill: Transfer Call

Purpose: Transfer the current voice call to a human agent when the automated system cannot handle the request.

Use when:
- The user explicitly asks to speak to a human or agent.
- The user is frustrated, angry, or repeats themselves multiple times.
- The request is too complex, sensitive, or falls outside the agent's capabilities.
- The agent encounters repeated tool errors or API failures.

Allowed tools:
- transfer_to_human

Steps:
1. Confirm with the user that they wish to be transferred to a human agent.
2. Collect a brief reason for the transfer from the conversation context (e.g., "user requested refund", "technical issue outside scope").
3. Call the `transfer_to_human` tool with the reason.
4. Inform the user of the transfer status (e.g., "Let me connect you now" or "An agent will be with you shortly").

Rules:
- Always confirm intent before transferring (never surprise the user).
- Do not transfer for trivial issues that the voice agent can resolve.
- If the transfer queue is full or unavailable, inform the user and offer a callback.
- Provide the reason string in a concise, professional format not exceeding 200 characters.
- Record the transfer reason for quality assurance but do not expose it to the user unless asked.
