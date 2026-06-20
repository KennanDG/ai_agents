# Coding Agent Desktop

React and TypeScript desktop shell for the coding agent. The UI uses Vite, Electron, Tailwind CSS, and Monaco Editor.

## Commands

```bash
npm install
npm run desktop:dev
npm run typecheck
npm run build
npm run desktop:build
```

The current UI uses typed mock session data. Replace `src/data/mockSession.ts` with an API client once the Python agent exposes an HTTP or WebSocket transport.
