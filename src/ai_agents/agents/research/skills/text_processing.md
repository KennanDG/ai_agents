# Skill: Text Processing

Purpose: Summarize, analyze, and transform text.

Use when:
- The user asks for a summary, keywords, or translation.
- The task requires condensing or interpreting text.

Steps:
1. Receive the input text.
2. Use summarize_text to produce a concise summary.
3. Use extract_keywords to identify key terms.
4. Use translate_text to convert text if needed.

Rules:
- Avoid modifying original text in-place without user consent.
- Do not summarize or translate secrets.
