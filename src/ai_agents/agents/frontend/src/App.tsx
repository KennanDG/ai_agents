import { useEffect, useMemo, useRef, useState } from "react";
import { ActivityBar } from "./components/ActivityBar";
import { DiffPanel } from "./components/DiffPanel";
import { OutputPanel } from "./components/OutputPanel";
import { Sidebar } from "./components/Sidebar";
import { TaskPanel } from "./components/TaskPanel";
import { fileChanges, messages } from "./data/mockSession";
import { createCodingAgentSocket, type CodingAgentServerEvent, } from "./lib/codingAgentSocket";

function App() {
  const [activePath, setActivePath] = useState(fileChanges[0].path);
  const [agentEvents, setAgentEvents] = useState<CodingAgentServerEvent[]>([]);
  const socketRef = useRef<ReturnType<typeof createCodingAgentSocket> | null>(null);

  const activeChange = useMemo(
    () => fileChanges.find((change) => change.path === activePath) ?? fileChanges[0],
    [activePath],
  );

  useEffect(() => {
    const apiBaseUrl = import.meta.env.VITE_AI_AGENTS_API_BASE ?? "http://localhost:8000";
    const apiKey = import.meta.env.VITE_AI_AGENTS_API_KEY;

    if (!apiKey) {
      console.warn("VITE_AI_AGENTS_API_KEY is not configured.");
      return;
    }

    const client = createCodingAgentSocket({
      apiBaseUrl,
      apiKey,
      onEvent: (event) => {
        setAgentEvents((current) => [...current, event]);
      },
      onOpen: () => {
        console.log("Coding agent socket connected.");
      },
      onClose: () => {
        console.log("Coding agent socket closed.");
      },
      onError: (event) => {
        console.error("Coding agent socket error.", event);
      },
    });

    socketRef.current = client;

    return () => {
      client.close();
      socketRef.current = null;
    };
  }, []);

  return (
    <main className="flex h-screen min-h-175 min-w-275 overflow-hidden bg-canvas text-ink">
      <ActivityBar />
      <Sidebar changes={fileChanges} activePath={activePath} onSelect={setActivePath} />
      <TaskPanel messages={messages} />
      <div className="flex min-w-0 flex-1 flex-col">
        <DiffPanel change={activeChange} />
        <OutputPanel />
      </div>
    </main>
  );
}

export default App;
