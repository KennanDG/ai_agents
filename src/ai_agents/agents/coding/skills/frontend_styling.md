# Skill: Style Frontend UI

Purpose: Build consistent, responsive, and accessible interfaces with the project's Tailwind design system.

Use when:
- The user asks to style a screen or component with Tailwind.
- The request concerns layout, visual hierarchy, responsiveness, theming, or accessibility.

Allowed tools:
- list_files
- robust_search
- read_file
- write_file
- run_command
- validate_tailwind_config
- audit_accessibility
- check_responsive_breakpoints

Steps:
1. Inspect global styles, theme tokens, and similar components before choosing classes.
2. Reuse theme colors, spacing, typography, and shared component classes.
3. Design keyboard, focus, hover, active, disabled, empty, and error states together.
4. Use semantic HTML and accessible names before adding ARIA attributes.
5. Check desktop layout at its declared minimum size and at narrower supported widths.
6. Run typecheck and build validation after styling changes.

Rules:
- Do not add one-off colors or spacing values when an existing token fits.
- Do not remove visible focus indicators.
- Do not encode meaning through color alone.
- Prefer Tailwind utilities and existing component classes over new inline styles.
- Respect `prefers-reduced-motion` for nonessential animation.
