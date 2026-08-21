import path from "node:path";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, shell } from "electron";

const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
const applicationRoot = path.join(currentDirectory, "..");
const developmentServerUrl = process.env.VITE_DEV_SERVER_URL;

function createWindow() {
  const window = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#090b10",
    title: "Coding Agent",
    webPreferences: {
      preload: path.join(currentDirectory, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  if (developmentServerUrl) {
    void window.loadURL(developmentServerUrl);
  } else {
    void window.loadFile(path.join(applicationRoot, "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// import path from "node:path";
// import net from "node:net";
// import { randomBytes } from "node:crypto";
// import {
//   existsSync,
//   mkdirSync,
// } from "node:fs";
// import {
//   spawn,
//   type ChildProcess,
// } from "node:child_process";
// import { fileURLToPath } from "node:url";

// import {
//   app,
//   BrowserWindow,
//   dialog,
//   ipcMain,
//   shell,
//   type OpenDialogOptions,
// } from "electron";


// const currentDirectory = path.dirname(
//   fileURLToPath(import.meta.url),
// );

// const applicationRoot = path.join(
//   currentDirectory,
//   "..",
// );

// const developmentServerUrl =
//   process.env.VITE_DEV_SERVER_URL;

// const BACKEND_HOST = "127.0.0.1";


// type DesktopDirectoryPickerOptions = {
//   title?: string;
//   defaultPath?: string;
// };


// type BackendInfo = {
//   baseUrl: string;
//   apiKey: string;
//   managed: boolean;
// };


// type BackendLaunchConfig = {
//   command: string;
//   args: string[];
//   cwd: string;
//   extraEnv?: NodeJS.ProcessEnv;
// };


// let backendProcess: ChildProcess | null = null;
// let backendInfo: BackendInfo | null = null;
// let applicationIsQuitting = false;


// /* -------------------------------------------------------------------------- */
// /*                              Local app paths                               */
// /* -------------------------------------------------------------------------- */

// function getDesktopDataPaths() {
//   /*
//    * userData resolves to an OS-specific, per-user directory.
//    *
//    * Windows:
//    *   %APPDATA%/<app-name>
//    *
//    * Linux:
//    *   ~/.config/<app-name>
//    *
//    * macOS:
//    *   ~/Library/Application Support/<app-name>
//    */
//   const userData = app.getPath("userData");

//   const memoryDir = path.join(
//     userData,
//     "memory",
//   );

//   const runtimeConfigPath = path.join(
//     userData,
//     "runtime-agent-config.json",
//   );

//   const githubWorkspaceRoot = path.join(
//     userData,
//     "github-workspaces",
//   );

//   mkdirSync(memoryDir, {
//     recursive: true,
//   });

//   mkdirSync(githubWorkspaceRoot, {
//     recursive: true,
//   });

//   return {
//     userData,
//     memoryDir,
//     runtimeConfigPath,
//     githubWorkspaceRoot,
//   };
// }


// /* -------------------------------------------------------------------------- */
// /*                            Development paths                               */
// /* -------------------------------------------------------------------------- */

// function findProjectRoot(): string {
//   const startingPoints = [
//     process.env.AI_AGENTS_PROJECT_ROOT,
//     process.cwd(),
//     applicationRoot,
//     app.getAppPath(),
//   ].filter(
//     (value): value is string =>
//       Boolean(value),
//   );

//   for (const startingPoint of startingPoints) {
//     let candidate = path.resolve(startingPoint);

//     while (true) {
//       const pyproject = path.join(
//         candidate,
//         "pyproject.toml",
//       );

//       const apiMain = path.join(
//         candidate,
//         "src",
//         "ai_agents",
//         "api",
//         "main.py",
//       );

//       if (
//         existsSync(pyproject) &&
//         existsSync(apiMain)
//       ) {
//         return candidate;
//       }

//       const parent = path.dirname(candidate);

//       if (parent === candidate) {
//         break;
//       }

//       candidate = parent;
//     }
//   }

//   throw new Error(
//     "Unable to locate the ai_agents project root. " +
//       "Set AI_AGENTS_PROJECT_ROOT explicitly.",
//   );
// }


// function resolveDevelopmentPython(
//   projectRoot: string,
// ): string {
//   if (process.env.AI_AGENTS_PYTHON) {
//     return process.env.AI_AGENTS_PYTHON;
//   }

//   const virtualEnvPython =
//     process.platform === "win32"
//       ? path.join(
//           projectRoot,
//           ".venv",
//           "Scripts",
//           "python.exe",
//         )
//       : path.join(
//           projectRoot,
//           ".venv",
//           "bin",
//           "python",
//         );

//   if (existsSync(virtualEnvPython)) {
//     return virtualEnvPython;
//   }

//   /*
//    * Final fallback. For packaged builds we never rely on a
//    * system Python installation.
//    */
//   return process.platform === "win32"
//     ? "python"
//     : "python3";
// }


// /* -------------------------------------------------------------------------- */
// /*                             Packaged backend                               */
// /* -------------------------------------------------------------------------- */

// function resolvePackagedBackend(): BackendLaunchConfig {
//   /*
//    * Optional escape hatch. Useful during packaging/testing.
//    */
//   const override =
//     process.env.AI_AGENTS_BACKEND_EXECUTABLE;

//   if (override) {
//     return {
//       command: override,
//       args: [],
//       cwd: path.dirname(override),
//     };
//   }

//   /*
//    * Recommended production layout:
//    *
//    * resources/
//    *   backend/
//    *     ai-agents-backend.exe     Windows
//    *     ai-agents-backend         macOS/Linux
//    */
//   const backendDir = path.join(
//     process.resourcesPath,
//     "backend",
//   );

//   const frozenExecutable =
//     process.platform === "win32"
//       ? path.join(
//           backendDir,
//           "ai-agents-backend.exe",
//         )
//       : path.join(
//           backendDir,
//           "ai-agents-backend",
//         );

//   if (existsSync(frozenExecutable)) {
//     return {
//       command: frozenExecutable,
//       args: [],
//       cwd: backendDir,
//     };
//   }

//   /*
//    * Also support a bundled Python runtime instead of a
//    * PyInstaller/Nuitka executable.
//    *
//    * resources/
//    *   backend/
//    *     python/
//    *     src/
//    */
//   const bundledPython =
//     process.platform === "win32"
//       ? path.join(
//           backendDir,
//           "python",
//           "python.exe",
//         )
//       : path.join(
//           backendDir,
//           "python",
//           "bin",
//           "python3",
//         );

//   const bundledSource = path.join(
//     backendDir,
//     "src",
//   );

//   if (
//     existsSync(bundledPython) &&
//     existsSync(bundledSource)
//   ) {
//     return {
//       command: bundledPython,
//       args: [
//         "-m",
//         "ai_agents.api.main",
//       ],
//       cwd: backendDir,
//       extraEnv: {
//         PYTHONPATH: bundledSource,
//       },
//     };
//   }

//   throw new Error(
//     [
//       "The packaged Python backend could not be found.",
//       `Expected: ${frozenExecutable}`,
//       "or a bundled Python runtime under resources/backend/python.",
//     ].join("\n"),
//   );
// }


// function resolveBackendLaunch(): BackendLaunchConfig {
//   if (app.isPackaged) {
//     return resolvePackagedBackend();
//   }

//   const projectRoot = findProjectRoot();
//   const python = resolveDevelopmentPython(
//     projectRoot,
//   );

//   return {
//     command: python,
//     args: [
//       "-m",
//       "ai_agents.api.main",
//     ],
//     cwd: projectRoot,
//     extraEnv: {
//       PYTHONPATH: [
//         path.join(
//           projectRoot,
//           "src",
//         ),
//         process.env.PYTHONPATH,
//       ]
//         .filter(Boolean)
//         .join(path.delimiter),

//       /*
//        * Development only.
//        *
//        * Do not ship your project .env inside the production
//        * Electron application.
//        */
//       ENV_FILE: path.join(
//         projectRoot,
//         ".env",
//       ),
//     },
//   };
// }


// /* -------------------------------------------------------------------------- */
// /*                                  Port                                      */
// /* -------------------------------------------------------------------------- */

// function reserveAvailablePort(): Promise<number> {
//   return new Promise((resolve, reject) => {
//     const server = net.createServer();

//     server.unref();

//     server.once(
//       "error",
//       reject,
//     );

//     server.listen(
//       {
//         host: BACKEND_HOST,
//         port: 0,
//         exclusive: true,
//       },
//       () => {
//         const address = server.address();

//         if (
//           !address ||
//           typeof address === "string"
//         ) {
//           server.close();

//           reject(
//             new Error(
//               "Unable to determine backend port.",
//             ),
//           );

//           return;
//         }

//         const port = address.port;

//         server.close((error) => {
//           if (error) {
//             reject(error);
//             return;
//           }

//           resolve(port);
//         });
//       },
//     );
//   });
// }


// /* -------------------------------------------------------------------------- */
// /*                           Backend health check                             */
// /* -------------------------------------------------------------------------- */

// function sleep(milliseconds: number) {
//   return new Promise<void>((resolve) => {
//     setTimeout(
//       resolve,
//       milliseconds,
//     );
//   });
// }


// async function waitForBackend(
//   baseUrl: string,
//   timeoutMs = 30_000,
// ): Promise<void> {
//   const deadline =
//     Date.now() + timeoutMs;

//   while (Date.now() < deadline) {
//     const controller =
//       new AbortController();

//     const timeout = setTimeout(
//       () => controller.abort(),
//       1_500,
//     );

//     try {
//       const response = await fetch(
//         `${baseUrl}/health`,
//         {
//           signal: controller.signal,
//         },
//       );

//       if (response.ok) {
//         clearTimeout(timeout);
//         return;
//       }
//     } catch {
//       // Backend is still starting.
//     } finally {
//       clearTimeout(timeout);
//     }

//     await sleep(250);
//   }

//   throw new Error(
//     `Python backend did not become ready within ${timeoutMs}ms.`,
//   );
// }


// /* -------------------------------------------------------------------------- */
// /*                          Python backend lifecycle                          */
// /* -------------------------------------------------------------------------- */

// async function startBackend(): Promise<BackendInfo> {
//   /*
//    * Allows development against a separately managed backend.
//    *
//    * Example:
//    * AI_AGENTS_BACKEND_URL=http://127.0.0.1:8000
//    */
//   const externalBackendUrl =
//     process.env.AI_AGENTS_BACKEND_URL?.trim();

//   if (externalBackendUrl) {
//     backendInfo = {
//       baseUrl:
//         externalBackendUrl.replace(/\/+$/, ""),
//       apiKey:
//         process.env.AI_AGENTS_API_KEY ?? "",
//       managed: false,
//     };

//     await waitForBackend(
//       backendInfo.baseUrl,
//     );

//     return backendInfo;
//   }

//   const port =
//     await reserveAvailablePort();

//   const apiKey = randomBytes(
//     32,
//   ).toString("hex");

//   const baseUrl =
//     `http://${BACKEND_HOST}:${port}`;

//   const dataPaths =
//     getDesktopDataPaths();

//   const launch =
//     resolveBackendLaunch();

//   const rendererOrigin =
//     developmentServerUrl
//       ? new URL(
//           developmentServerUrl,
//         ).origin
//       : "null";

//   const childEnv: NodeJS.ProcessEnv = {
//     ...process.env,
//     ...launch.extraEnv,

//     /*
//      * FastAPI binding
//      */
//     AI_AGENTS_HOST: BACKEND_HOST,
//     AI_AGENTS_PORT: String(port),

//     /*
//      * Per-launch local authentication.
//      */
//     AI_AGENTS_API_KEY: apiKey,

//     /*
//      * Electron renderer origin(s).
//      */
//     AI_AGENTS_ALLOWED_ORIGINS:
//       rendererOrigin,

//     /*
//      * SQLite + FastEmbed persistent memory.
//      *
//      * coding_agent_settings.py derives:
//      *
//      *   checkpoints.sqlite3
//      *   store.sqlite3
//      *   fastembed-cache/
//      *
//      * underneath this directory.
//      */
//     CODING_AGENT_MEMORY_DIR:
//       dataPaths.memoryDir,

//     CODING_AGENT_MEMORY_ENABLED:
//       "true",

//     CODING_AGENT_MEMORY_SETUP:
//       "true",

//     CODING_AGENT_MEMORY_SEMANTIC:
//       "true",

//     /*
//      * Other application state should not live relative to
//      * the installation directory either.
//      */
//     AI_AGENTS_RUNTIME_CONFIG_PATH:
//       dataPaths.runtimeConfigPath,

//     GITHUB_WORKSPACE_ROOT:
//       dataPaths.githubWorkspaceRoot,

//     /*
//      * Safe LangGraph checkpoint serialization.
//      */
//     LANGGRAPH_STRICT_MSGPACK:
//       "true",

//     /*
//      * Desktop logging/process behavior.
//      */
//     PYTHONUNBUFFERED:
//       "1",

//     PYTHONUTF8:
//       "1",

//     /*
//      * Keep the desktop backend free from the existing
//      * Postgres/Qdrant RAG stack until that subsystem is
//      * migrated separately.
//      */
//     AI_AGENTS_ENABLE_RAG:
//       "false",
//   };

//   console.log(
//     `[backend] Starting ${launch.command} ${launch.args.join(" ")}`,
//   );

//   console.log(
//     `[backend] URL: ${baseUrl}`,
//   );

//   console.log(
//     `[backend] memory: ${dataPaths.memoryDir}`,
//   );

//   backendProcess = spawn(
//     launch.command,
//     launch.args,
//     {
//       cwd: launch.cwd,
//       env: childEnv,
//       windowsHide: true,
//       stdio: [
//         "ignore",
//         "pipe",
//         "pipe",
//       ],
//     },
//   );

//   backendProcess.stdout?.on(
//     "data",
//     (chunk) => {
//       process.stdout.write(
//         `[python] ${chunk}`,
//       );
//     },
//   );

//   backendProcess.stderr?.on(
//     "data",
//     (chunk) => {
//       process.stderr.write(
//         `[python] ${chunk}`,
//       );
//     },
//   );

//   await new Promise<void>(
//     (resolve, reject) => {
//       if (!backendProcess) {
//         reject(
//           new Error(
//             "Backend process was not created.",
//           ),
//         );
//         return;
//       }

//       backendProcess.once(
//         "spawn",
//         () => resolve(),
//       );

//       backendProcess.once(
//         "error",
//         reject,
//       );
//     },
//   );

//   backendProcess.on(
//     "exit",
//     (code, signal) => {
//       console.log(
//         `[backend] exited code=${code} signal=${signal}`,
//       );

//       backendProcess = null;

//       if (
//         !applicationIsQuitting &&
//         backendInfo
//       ) {
//         backendInfo = null;

//         dialog.showErrorBox(
//           "Backend stopped",
//           "The local Python backend stopped unexpectedly. " +
//             "Restart the application to continue.",
//         );
//       }
//     },
//   );

//   try {
//     await waitForBackend(
//       baseUrl,
//     );
//   } catch (error) {
//     stopBackend();
//     throw error;
//   }

//   backendInfo = {
//     baseUrl,
//     apiKey,
//     managed: true,
//   };

//   return backendInfo;
// }


// function stopBackend() {
//   const child = backendProcess;

//   backendProcess = null;
//   backendInfo = null;

//   if (
//     !child ||
//     child.killed
//   ) {
//     return;
//   }

//   try {
//     child.kill();
//   } catch (error) {
//     console.error(
//       "[backend] failed to terminate:",
//       error,
//     );
//   }
// }


// /* -------------------------------------------------------------------------- */
// /*                                  IPC                                       */
// /* -------------------------------------------------------------------------- */

// function registerDesktopDirectoryPicker() {
//   ipcMain.handle(
//     "desktop:select-directory",
//     async (
//       event,
//       options?: DesktopDirectoryPickerOptions,
//     ) => {
//       const owner =
//         BrowserWindow.fromWebContents(
//           event.sender,
//         );

//       const dialogOptions: OpenDialogOptions = {
//         title:
//           options?.title ??
//           "Select repository root",

//         defaultPath:
//           options?.defaultPath,

//         properties: [
//           "openDirectory",
//           "createDirectory",
//         ],
//       };

//       const result = owner
//         ? await dialog.showOpenDialog(
//             owner,
//             dialogOptions,
//           )
//         : await dialog.showOpenDialog(
//             dialogOptions,
//           );

//       if (result.canceled) {
//         return null;
//       }

//       return (
//         result.filePaths[0] ??
//         null
//       );
//     },
//   );
// }


// function registerBackendIpc() {
//   ipcMain.handle(
//     "desktop:get-backend-info",
//     () => {
//       if (!backendInfo) {
//         throw new Error(
//           "Python backend is not ready.",
//         );
//       }

//       return backendInfo;
//     },
//   );
// }


// /* -------------------------------------------------------------------------- */
// /*                              Browser window                                */
// /* -------------------------------------------------------------------------- */

// function createWindow() {
//   const window = new BrowserWindow({
//     width: 1440,
//     height: 900,

//     minWidth: 1100,
//     minHeight: 700,

//     backgroundColor: "#090b10",
//     title: "Coding Agent",

//     webPreferences: {
//       preload: path.join(
//         currentDirectory,
//         "preload.mjs",
//       ),

//       contextIsolation: true,
//       nodeIntegration: false,
//       sandbox: true,
//     },
//   });

//   window.webContents.setWindowOpenHandler(
//     ({ url }) => {
//       void shell.openExternal(url);

//       return {
//         action: "deny",
//       };
//     },
//   );

//   if (developmentServerUrl) {
//     void window.loadURL(
//       developmentServerUrl,
//     );
//   } else {
//     void window.loadFile(
//       path.join(
//         applicationRoot,
//         "dist",
//         "index.html",
//       ),
//     );
//   }
// }


// /* -------------------------------------------------------------------------- */
// /*                           Electron application                             */
// /* -------------------------------------------------------------------------- */

// app.whenReady().then(
//   async () => {
//     registerDesktopDirectoryPicker();
//     registerBackendIpc();

//     try {
//       await startBackend();
//     } catch (error) {
//       const message =
//         error instanceof Error
//           ? error.message
//           : String(error);

//       console.error(
//         "[backend] startup failed:",
//         error,
//       );

//       dialog.showErrorBox(
//         "Backend startup failed",
//         message,
//       );

//       app.quit();
//       return;
//     }

//     createWindow();

//     app.on(
//       "activate",
//       () => {
//         if (
//           BrowserWindow.getAllWindows()
//             .length === 0
//         ) {
//           createWindow();
//         }
//       },
//     );
//   },
// );


// app.on(
//   "before-quit",
//   () => {
//     applicationIsQuitting = true;
//     stopBackend();
//   },
// );


// app.on(
//   "window-all-closed",
//   () => {
//     if (
//       process.platform !== "darwin"
//     ) {
//       app.quit();
//     }
//   },
// );



