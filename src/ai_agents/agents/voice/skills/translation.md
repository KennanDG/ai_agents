# Skill: Translate Text

Purpose: Translate user-provided text from one language to another.

Use when:
- The user asks to translate a phrase, sentence, or document.
- The user switches languages or specifies a target language.

Allowed tools:
- translate_text (or similar language translation API)

Steps:
1. Identify the text to be translated and the source and target languages.
2. If languages are ambiguous, ask the user for clarification.
3. Call the translation tool with the text and target language.
4. Return the translated text in a clear, natural format.

Rules:
- Preserve the original meaning and tone as much as possible.
- Handle special characters, emojis, and markdown formatting gracefully.
- If the text is too long, summarize or suggest breaking it into parts.
- Always confirm the target language if not explicitly stated.
- Do not invent translations; rely on the translation tool.
