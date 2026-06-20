import { useMemo, useState } from "react";
import { ActivityBar } from "./components/ActivityBar";
import { DiffPanel } from "./components/DiffPanel";
import { OutputPanel } from "./components/OutputPanel";
import { Sidebar } from "./components/Sidebar";
import { TaskPanel } from "./components/TaskPanel";
import { fileChanges, messages } from "./data/mockSession";

function App() {
  const [activePath, setActivePath] = useState(fileChanges[0].path);
  const activeChange = useMemo(
    () => fileChanges.find((change) => change.path === activePath) ?? fileChanges[0],
    [activePath],
  );

  return (
    <main className="flex h-screen min-h-[700px] min-w-[1100px] overflow-hidden bg-canvas text-ink">
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
