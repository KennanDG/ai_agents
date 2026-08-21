import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("desktop", {
  platform: process.platform,
});

// import {
//   contextBridge,
//   ipcRenderer,
// } from "electron";


// type DesktopDirectoryPickerOptions = {
//   title?: string;
//   defaultPath?: string;
// };


// export type DesktopBackendInfo = {
//   baseUrl: string;
//   apiKey: string;
//   managed: boolean;
// };


// contextBridge.exposeInMainWorld(
//   "desktop",
//   {
//     platform: process.platform,

//     selectDirectory: (
//       options?: DesktopDirectoryPickerOptions,
//     ): Promise<string | null> =>
//       ipcRenderer.invoke(
//         "desktop:select-directory",
//         options,
//       ),

//     getBackendInfo:
//       (): Promise<DesktopBackendInfo> =>
//         ipcRenderer.invoke(
//           "desktop:get-backend-info",
//         ),
//   },
// );

